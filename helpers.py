import numpy as np
from scipy.sparse import eye as speye
import math

import os, h5py

def load_Q_dataset(power, labels, base_path="/disk/hyk049/DHM_new_1Dcenter"):
    """
    Load Q datasets for given power and labels.

    Parameters
    ----------
    power : str
        Example: '0p10'
    labels : list of str
        Example: ['a','b','c',...]
    base_path : str
        Root directory

    Returns
    -------
    t : ndarray
    x : ndarray
    Q : dict
        Dictionary of Q matrices keyed by label
    nx : int
    """
    
    base = os.path.join(base_path, power)

    def h5read(fname, dset):
        with h5py.File(fname, "r") as f:
            return np.array(f[dset])

    # Use first label to load t, x
    first_file = os.path.join(base, f"Q_1D_{power}vpp_{labels[0]}.h5")
    t = h5read(first_file, "/t")
    x = h5read(first_file, "/x")
    nx = len(x)

    # Load all Q
    Q = {}
    for lbl in labels:
        fname = os.path.join(base, f"Q_1D_{power}vpp_{lbl}.h5")
        Q[lbl] = h5read(fname, "/Q_1D")

    return Q, t, x, nx

def preprocess_Q(Q_dict, t, labels, split_size=5760):
    """
    Preprocess Q data:
    - split into segments
    - concatenate across realizations
    - compute mean field

    Parameters
    ----------
    Q_dict : dict
        {label: Q matrix (nx, nt)}
    t : ndarray
        time vector
    labels : list
        list of labels to include
    split_size : int

    Returns
    -------
    Q_split : dict
        {label: (nx, num_segs, split_size)}
    Qstate_all : ndarray
        (nx, total_segs, split_size)
    X_mean : ndarray
        (nx, split_size)
    tt : ndarray
        truncated time vector
    """

    nt = len(t)
    num_segs = nt // split_size
    tt = t[:split_size]

    def split_Q(Qi):
        Qi = Qi[:, :num_segs * split_size]
        nx = Qi.shape[0]
        return Qi.reshape(nx, num_segs, split_size)

    # Split all
    Q_split = {lbl: split_Q(Q_dict[lbl]) for lbl in labels}

    # Concatenate across labels (segments axis = 1)
    Qstate_all = np.concatenate(
        [Q_split[lbl] for lbl in labels],
        axis=1
    )

    # Mean over segments
    X_mean = np.mean(Qstate_all, axis=1)

    return Q_split, Qstate_all, X_mean, tt


def page_norm(A):
    """
    Frobenius norm over all entries and pages.
    Equivalent to MATLAB: norm(A(:))
    """
    return np.linalg.norm(A.ravel())

def page_cov(x, transpose_pages=False):
    """
    Compute covariance matrix pagewise.

    Parameters
    ----------
    x : ndarray, shape (nObs, nVar, nPages) or (nVar, nObs, nPages)
    transpose_pages : bool
        If True, swap first two axes before computing covariance.

    Returns
    -------
    C : ndarray, shape (nVar, nVar, nPages)
        Pagewise covariance matrices.
    """
    if transpose_pages:
        x = np.transpose(x, (1, 0, 2))

    nObs, nVar, nPages = x.shape
    C = np.zeros((nVar, nVar, nPages))

    for i in range(nPages):
        Xi = x[:, :, i]                       # (nObs, nVar)
        Xi = Xi - Xi.mean(axis=0, keepdims=True)

        if nObs > 1:
            C[:, :, i] = (Xi.T @ Xi) / (nObs - 1)
        else:
            C[:, :, i] = np.zeros((nVar, nVar))

    return C

def central_finite_differences(x, h, ord=2, axis=1):
    """
    Compute central finite differences of x along a given axis.

    Parameters
    ----------
    x : ndarray
        Input array.
    h : float
        Time step.
    ord : int, optional
        Accuracy: 2, 4, 6, 8 (default=2).
    axis : int, optional
        Axis along which to compute derivative (default=1, 0-indexed in Python).

    Returns
    -------
    y : ndarray
        Interior points of x where derivative is defined.
    dxdt : ndarray
        Central finite difference derivative along axis.
    ind : ndarray
        Valid indices along the axis used.
    """

    # -----------------------------
    # Coefficients for central differences
    # -----------------------------
    if ord == 2:
        indx = np.array([-1, 0, 1])
        coef = np.array([-0.5, 0.0, 0.5]) / h
    elif ord == 4:
        indx = np.array([-2, -1, 0, 1, 2])
        coef = np.array([1/12, -2/3, 0, 2/3, -1/12]) / h
    elif ord == 6:
        indx = np.array([-3, -2, -1, 0, 1, 2, 3])
        coef = np.array([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60]) / h
    elif ord == 8:
        indx = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
        coef = np.array([1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280]) / h
    else:
        raise ValueError("acc can only be 2, 4, 6, or 8!")

    # -----------------------------
    # Determine interior indices
    # -----------------------------
    p = len(coef)
    n = x.shape[axis]
    start = (p - 1) // 2
    end = n - (p - 1) // 2
    ind = np.arange(start, end)  # valid indices

    # -----------------------------
    # Allocate output lists
    # -----------------------------
    dxdt_list = []
    y_list = []

    # -----------------------------
    # Iterate over valid indices
    # -----------------------------
    for ii in ind:
        # Slice array around current index along given axis
        slicer = [slice(None)] * x.ndim
        values = []
        for k, offset in enumerate(indx):
            slicer[axis] = ii + offset
            values.append(coef[k] * x[tuple(slicer)])

        dX = sum(values)

        # Current value
        slicer[axis] = ii
        Xii = x[tuple(slicer)]

        dxdt_list.append(dX)
        y_list.append(Xii)

    # Stack along the axis
    dxdt = np.stack(dxdt_list, axis=axis)
    y = np.stack(y_list, axis=axis)

    return y, dxdt, ind


def regularizer(r, lam):
    """
    Construct a diagonal regularizer matrix.

    Parameters
    ----------
    r : int
        Reduced dimension.
    lam : float
        Regularization parameter (lambda).

    Returns
    -------
    Reg : ndarray, shape (r, r)
        Diagonal regularizer matrix.
    """

    # Allocate vector
    Reg_vec = np.zeros(r)

    # Regularizer for linear operator A
    Reg_vec[:r] = lam

    # Construct diagonal matrix
    Reg = np.diag(Reg_vec)

    return Reg


def infer_drift(E_train, h, isbilinear, s, reg=None):
    """
    Infer the drift operators for the reduced-order model (OpInf).

    Parameters
    ----------
    E_train : ndarray, shape (r, s)
        Expected value of the reduced states across samples.
    h : float
        Time step.
    isbilinear : bool
        Whether the system has bilinear terms.
    s : int
        Number of time snapshots.
    reg : float or None
        Regularization parameter. If None, no regularization is applied.

    Returns
    -------
    Ehat : ndarray, shape (r, r)
        Mass matrix (identity in our examples).
    Ahat : ndarray, shape (r, r)
        Linear reduced operator.
    Nhat : ndarray, shape (r, ?)
        Bilinear reduced operator (zeros if isbilinear=False).
    """

    r = E_train.shape[0]
    D = np.empty((0, 0))
    rhs = np.empty((0, 0))

    m = 1
    # m = u_train{1}.shape[0]
    # p = numel(E_train);
    # u = u_train{ii};

    # Central finite difference (accuracy 2)
    # Returns Er (states), Er_dot (time derivatives), ind (indices used)
    Er, Er_dot, ind = central_finite_differences(E_train, h, ord=2, axis=1)
    
    # Er_dot = np.zeros_like(E_train)
    # for j in range(E_train.shape[0]):
    #     Er_dot[j, :] = ctr_FD(E_train[j, :], h, order=2)
    # Er = E_train
    
    D = Er
    rhs = Er_dot

    # print(f"cond(D) = {np.linalg.cond(D):.4e}")


    # Solve least-squares problem
    if reg is not None:
        Gamma = regularizer(r, reg)        # user-defined
        D_modified = D.T @ D + Gamma.T @ Gamma
        rhs_modified = D.T @ rhs.T          # note: rhs.T to match MATLAB dimensions
        ops = np.linalg.solve(D_modified, rhs_modified)
        ops = ops.T
        Ahat = ops[:r, :r]
    else:
        # Non-regularized least-squares
        # ops = rhs / D  --> MATLAB left-division
        ops = np.linalg.lstsq(D.T, rhs.T, rcond=None)[0].T
        Ahat = ops[:r, :r]

    # Bilinear term
    if isbilinear:
        Nhat = ops[:, r+m:]      # m must be defined or passed if using inputs
    else:
        Nhat = np.zeros_like(Ahat)

    # Mass matrix
    Ehat = speye(r).toarray()   # dense identity matrix

    return Ehat, Ahat, Nhat


from scipy.signal import savgol_filter

def infer_diffusion(C_train, h, Ahat, Nhat=None):
    """
    Infer the diffusion operator for the reduced-order model (OpInf).

    Parameters
    ----------
    C_train : ndarray, shape (r, r, s)
        Covariance of reduced states at each snapshot.
    h : float
        Sampling time step.
    Ahat : ndarray, shape (r, r)
        Linear reduced operator from drift inference.
    Nhat : ndarray, optional
        Bilinear reduced operator from drift inference (unused here).

    Returns
    -------
    Mhat : ndarray, shape (r, d)
        Reduced diffusion matrix (truncated).
    Khat : ndarray, shape (d, d)
        Correlation matrix (identity).
    """

    r = Ahat.shape[0]
    Hhat = np.zeros_like(Ahat)

    # Central finite differences along the 3rd axis (time)
    Cr_dot = savgol_filter(C_train, window_length=7, polyorder=3, deriv=1, delta=h, axis=2)
    Cr = C_train
    ind = Cr.shape[2]
    
    # Cr, Cr_dot, ind = central_finite_differences(C_train, h, ord=2, axis=2)


    # Loop over valid indices
    for jj in range(ind):
    # for jj in range(len(ind)):
        Psi_hat = Ahat
        Clyap = Psi_hat @ Cr[:, :, jj]

        # Increment for diffusion contribution
        incre = Cr_dot[:, :, jj] - Clyap - Clyap.T
        
        Hhat += incre / ind
        # Hhat += incre / len(ind)   # mean over indices

    # Ensure Hhat is symmetric
    Hhat = 0.5 * (Hhat + Hhat.T)

    # Eigen-decomposition
    HS_vals, HU = np.linalg.eigh(Hhat)  # HU columns = eigenvectors
    # Sort eigenvalues descending
    idx = np.argsort(HS_vals)[::-1]
    HS_vals = HS_vals[idx]
    HU = HU[:, idx]

    # Truncate based on tolerance (0.1% of max eigenvalue)
    tol = np.max(HS_vals) / 1000
    d_candidates = np.where(HS_vals >= tol)[0]
    if len(d_candidates) > 0:
        d = d_candidates[-1] + 1  # +1 for Python indexing
    else:
        d = 0

    # Construct Mhat
    if d > 0:
        Mhat = HU[:, :d] @ np.diag(np.sqrt(HS_vals[:d]))
    else:
        Mhat = np.zeros((r, 0))

    # Correlation matrix
    Khat = np.eye(d)

    return Mhat, Khat



def compute_model(f, V, xr0_test, s, batch_size, L_test):
    """
    Compute mean and covariance of ROM/FOM trajectories using batch processing.

    Parameters
    ----------
    f : callable
        Euler-Maruyama step function: f(x0, L) -> batch of trajectories.
    V : ndarray, shape (n, r)
        POD basis (identity if FOM).
    xr0_test : ndarray, shape (r,)
        Initial condition of the test.
    s : int
        Number of snapshots/time points.
    batch_size : int
        Number of trajectories per batch.
    L_test : int
        Total number of noise samples.

    Returns
    -------
    E_test : ndarray, shape (r, s)
        Mean of the trajectories.
    C_test : ndarray, shape (r, r, s)
        Covariance of the trajectories.
    f1 : float
        Second moment of X(end): ||X(end)||_2^2
    f2 : float
        Mean of X(T)^3 * exp(X(T))
    """

    r = xr0_test.shape[0]

    # Determine number of batches
    num_batches = math.ceil(L_test / batch_size)
    if L_test <= batch_size:
        num_batches = 1
        batch_size = L_test

    # Initialize outputs
    E_test = np.zeros((r, s))
    C_test = np.zeros((r, r, s))
    f1 = 0.0
    f2 = 0.0

    # Loop over batches
    for batch in range(num_batches):
        Nb = batch_size
        if batch == num_batches - 1:  # last batch
            Nb = L_test - batch * batch_size

        # Compute statistics for one batch
        # estimate() should return (E_temp, C_temp, f1_temp, f2_temp)
        E_temp, C_temp, f1_temp, f2_temp = estimate(f, V, xr0_test, s, Nb)

        # Accumulate batch mean
        E_test += (Nb / L_test) * E_temp

        # Accumulate second moment for covariance
        for k in range(s):
            C_test[:, :, k] += (Nb / L_test) * (C_temp[:, :, k] + np.outer(E_temp[:, k], E_temp[:, k]))

    # Finalize covariance by subtracting mean outer product
    for k in range(s):
        C_test[:, :, k] -= np.outer(E_test[:, k], E_test[:, k])

    return E_test, C_test, f1, f2


def estimate(f, Vr, xr0, s, L):
    """
    Estimate empirical mean, covariance, and diagnostics from stochastic simulations.

    Parameters
    ----------
    f : callable
        Euler-Maruyama step function: f(x0, L) -> simulated trajectories
    Vr : ndarray, shape (n, r)
        POD basis of the total snapshot (used for reconstruction)
    xr0 : ndarray, shape (r,)
        Initial condition
    s : int
        Number of time steps
    L : int
        Number of noise realizations

    Returns
    -------
    E_emp : ndarray, shape (r, s)
        Empirical mean over noise realizations
    C_emp : ndarray, shape (r, r, s)
        Empirical covariance of reduced states
    f1 : float
        Second moment of X(end): ||X(end)||_2^2
    f2 : float
        Mean of X(T)^3 / exp(X(T))
    """

    # Run L stochastic simulations
    # Xr shape: (r, L, s)
    Xr = stepSDE(f, xr0, s, L)

    # Empirical mean over L realizations
    E_emp = np.mean(Xr, axis=1)  # shape (r, s)

    # Empirical covariance
    C_emp = page_cov(Xr, transpose_pages=True)  # shape (r, r, s)

    # Diagnostics at final time
    X_T_recon = Vr @ Xr[:, :, -1]  # shape (n, L)

    # f1: second moment of final state
    # f1 = np.mean(np.linalg.norm(X_T_recon, axis=0)**2)

    # f2: mean over elements of X(T)^3 / exp(X(T))
    # f2 = np.mean(X_T_recon**3 / np.exp(X_T_recon))
    f1, f2 = 0, 0

    return E_emp, C_emp, f1, f2


def stepSDE(f, x0, s, L):
    """
    Simulate stochastic trajectories using Euler-Maruyama steps.

    Parameters
    ----------
    f : callable
        Euler-Maruyama step function: f(x0, L) -> next state(s), shape (r, L)
    x0 : ndarray, shape (r,)
        Initial condition
    s : int
        Total number of time steps
    L : int
        Number of noise realizations

    Returns
    -------
    Xr : ndarray, shape (r, L, s)
        Simulated trajectories across L noise realizations and s time steps
    """

    r = x0.shape[0]
    Xr = np.zeros((r, L, s))

    # First step
    Xr[:, :, 0] = f(x0, L)  # shape (r, L)
    
    # Remaining steps
    for ii in range(1, s):
        Xr[:, :, ii] = f(Xr[:, :, ii-1], L)

    return Xr


def regularizer_u(r, lam, isbilinear):
    """
    Construct regularizer diagonal matrix.

    Parameters
    ----------
    r : int
        Reduced dimension
    lam : array-like (length 2 or 3)
        Regularization parameters [lambda_A, lambda_B, (lambda_N)]
    isbilinear : bool
        Whether bilinear term is included

    Returns
    -------
    Reg : (d, d) ndarray
        Diagonal regularization matrix
    """

    m = 1

    if isbilinear:
        d = r + m + r * m
    else:
        d = r + m

    Reg_vec = np.zeros(d)

    # A block
    Reg_vec[:r] = lam[0]

    # B block
    Reg_vec[r] = lam[1]

    # N block (bilinear)
    if isbilinear:
        Reg_vec[r + 1:] = lam[2]

    Reg = np.diag(Reg_vec)

    return Reg


def infer_drift_u(E_train, u_train, h, isbilinear, regs=None):
    """
    Infer the drift operators for the reduced-order model (OpInf).

    Parameters
    ----------
    E_train : ndarray, shape (r, s)
        Expected value of the reduced states across samples.
    h : float
        Time step.
    isbilinear : bool
        Whether the system has bilinear terms.
    s : int
        Number of time snapshots.
    reg : float or None
        Regularization parameter. If None, no regularization is applied.

    Returns
    -------
    Ehat : ndarray, shape (r, r)
        Mass matrix (identity in our examples).
    Ahat : ndarray, shape (r, r)
        Linear reduced operator.
    Nhat : ndarray, shape (r, ?)
        Bilinear reduced operator (zeros if isbilinear=False).
    """

    r = E_train.shape[0]
    
    # m = len(u_train)
    m = 1
    u = u_train

    # Central finite difference (accuracy 2)
    # Returns Er (states), Er_dot (time derivatives), ind (indices used)
    Er, Er_dot, ind = central_finite_differences(E_train, h, ord=2, axis=1)
    
    # D = Er
    # rhs = Er_dot

    K = len(ind)
    
    if isbilinear:
        col_dim = r + m + r*m
    else:
        col_dim = r + m

    if isbilinear:
        # u: (m, K)
        # Er: (r, K)

        kron_block = np.zeros((r * m, K))

        for jj in range(K):
            # kron_block[:, jj] = np.kron(u[jj], Er[:, jj])
            # kron_block[:, jj] = np.kron(u[:, jj], Er[:, jj])
            kron_block[:, jj] = u[jj] * Er[:, jj]

        D = np.vstack([
            Er[:, :K],
            u[:K],
            kron_block
        ])
        rhs = Er_dot[:, :K]
        
    else:
        D = np.vstack([
            Er[:, :K],
            u[:, :K]
        ])
        rhs = Er_dot[:, :K]
        

    # Solve least-squares problem
    Dt = D.T
    Rt = rhs.T
    if regs is not None:
        Gamma = regularizer_u(r, regs, isbilinear)        # user-defined
        D_modified = Dt.T @ Dt + Gamma.T @ Gamma
        rhs_modified = D @ Rt          # note: rhs.T to match MATLAB dimensions
        ops = np.linalg.solve(D_modified, rhs_modified)
        ops = ops.T
    else:
        # Non-regularized least-squares
        # ops = rhs / D  --> MATLAB left-division
        ops = np.linalg.lstsq(Dt, Rt, rcond=None)[0].T
    
    # Extract operators
    Ahat = ops[:, :r]
    Bhat = ops[:, r:r + m]
    
    # Bilinear term
    if isbilinear:
        Nhat = ops[:, r+m:]      # m must be defined or passed if using inputs
    else:
        Nhat = np.zeros((ops.shape[0], 0))

    # Mass matrix
    Ehat = speye(r).toarray()   # dense identity matrix

    return Ehat, Ahat, Bhat, Nhat


def infer_diffusion_u(C_train, u_train, h, Ahat, Nhat, lam):
    """
    Infer diffusion operator from covariance dynamics.
    """

    # Cr, Cr_dot, ind = central_finite_differences(C_train, h, ord=2, axis=2)
    # T = len(ind)
    
    # Central finite differences along the 3rd axis (time)
    Cr_dot = savgol_filter(C_train, window_length=7, polyorder=3, deriv=1, delta=h, axis=2)
    Cr = C_train
    ind = Cr.shape[2]
    T = ind

    n = Ahat.shape[0]
    Hhat = np.zeros((n, n))

    # Build diffusion estimate
    for jj in range(T):

        # MATLAB: u_train(:, jj) or u_train(jj) ambiguity
        u_j = u_train[:, jj] if u_train.ndim > 1 else u_train[jj]

        Psi_hat = Ahat + Nhat * u_j   # assumes scalar u_j (as in MATLAB broadcast)
        Clyap = Psi_hat @ Cr[:, :, jj]

        incre = Cr_dot[:, :, jj] - (Clyap + Clyap.T)

        Hhat += incre 
        
    # Regularization
    # Hhat = Hhat / (T * lam)
    Hhat = Hhat / (T + lam)

    # Symmetrize
    Hhat = 0.5 * (Hhat + Hhat.T)

    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(Hhat)

    # sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    HS = np.diag(eigvals)
    HU = eigvecs

    # truncation threshold
    tol = np.max(eigvals) / 1e3

    valid = np.where(eigvals >= tol)[0]
    d = valid[-1] + 1 if len(valid) > 0 else 0

    # reconstruct diffusion factor
    if d > 0:
        Mhat = HU[:, :d] @ np.sqrt(HS[:d, :d])
    else:
        Mhat = np.zeros((n, 0))
    
    Khat = np.eye(d)

    return Mhat, Khat


def compute_model_u(f, V, xr0_test, u_test, batch_size, L_test):
    """
    Compute mean and covariance of FOM/ROM states with input u_test.

    Parameters
    ----------
    f : function
        Step function: f(x0, u, L) -> Euler-Maruyama simulation
    V : ndarray
        POD basis (n x n)
    xr0_test : ndarray
        Initial condition (r x 1)
    u_test : ndarray
        Input signal (m x s)
    batch_size : int
        Number of trajectories per batch
    L_test : int
        Total number of noise samples

    Returns
    -------
    E_test : ndarray
        Mean of the states (r x s)
    C_test : ndarray
        Covariance of the states (r x r x s)
    """

    r = xr0_test.shape[0]
    # m, s = u_test.shape
    s = len(u_test)

    # Determine number of batches
    num_batches = math.ceil(L_test / batch_size)
    if L_test <= batch_size:
        num_batches = 1
        batch_size = L_test

    # Initialize accumulators
    E_test = np.zeros((r, s))
    C_test = np.zeros((r, r, s))

    for batch in range(num_batches):
        # Last batch might be smaller
        Nb = batch_size
        if batch == num_batches - 1:
            Nb = L_test - (batch) * batch_size

        # Compute statistics for one batch
        E_temp, C_temp = estimate_u(f, xr0_test, u_test, Nb)

        # Accumulate mean
        E_test += (Nb / L_test) * E_temp

        # Accumulate second moments
        for k in range(s):
            # outer product of mean vector
            C_test[:, :, k] += (Nb / L_test) * (C_temp[:, :, k] + np.outer(E_temp[:, k], E_temp[:, k]))
            
    # Subtract mean outer product to get covariance
    for k in range(s):
        C_test[:, :, k] -= np.outer(E_test[:, k], E_test[:, k])

    return E_test, C_test


def estimate_u(f, xr0, u, L):
    """
    Empirical mean and covariance from stochastic simulations.
    """

    # Run stochastic simulations
    # Xr shape: (r, L, s)
    Xr = stepSDE_u(f, xr0, u, L)

    # Mean (over noise samples)
    E_empirical = np.mean(Xr, axis=1)   # shape: (r, s)

    # Empirical covariance
    C_empirical = page_cov(Xr, transpose_pages=True)  # shape (r, r, s)

    return E_empirical, C_empirical


def stepSDE_u(f, x0, u, L):
    """
    Run stochastic Euler–Maruyama steps over time with input u.

    Parameters
    ----------
    f : function
        EulerMaruyamaStep-like function:
        f(x, u_t, L) -> (r, L) samples

    x0 : ndarray
        Initial condition (r,)

    u : ndarray
        Input signal (m, s) or (s,) depending on model

    L : int
        Number of noise realizations

    Returns
    -------
    Xr : ndarray
        Shape (r, L, s)
    """

    s = len(u)
    r = x0.shape[0]

    Xr = np.zeros((r, L, s))

    # First time step
    Xr[:, :, 0] = f(x0, u[:, 0] if u.ndim > 1 else u[0], L)

    # Time stepping
    for ii in range(1, s):
        u_prev = u[:, ii-1] if u.ndim > 1 else u[ii-1]
        Xr[:, :, ii] = f(Xr[:, :, ii-1], u_prev, L)

    return Xr


from sympy import symbols, factorial, Matrix, Rational
def finite_diff_coeffs(x_vals, x0, derivative_order):
    """
    x_vals: list of grid points (e.g., [0, 1, 2, 3, 4, 5, 6])
    x0: the point at which the derivative is to be approximated (e.g., 1 for coeffs1)
    derivative_order: which derivative (1 for first derivative)
    """
    m = len(x_vals)
    # k outer loop, x inner loop
    A = Matrix([[Rational((x - x0)**k, factorial(k)) for x in x_vals] for k in range(m)])
    b = Matrix([1 if k == derivative_order else 0 for k in range(m)])
    coeffs = A.LUsolve(b)
    return np.array(coeffs, dtype=np.float64)

def ctr_FD_tensor(X, h, order, axis=-1):
    """
    High-order central FD along given axis (vectorized).
    """

    half = order // 2
    offsets = np.arange(-half, half + 1)

    # central coefficients (scaled properly)
    coeffs = finite_diff_coeffs(offsets * h, 0, 1)
    coeffs = np.array(coeffs, dtype=float)

    # valid indices
    n = X.shape[axis]
    ind = np.arange(half, n - half)

    # slice central region
    slices = [slice(None)] * X.ndim
    slices[axis] = ind
    Xc = X[tuple(slices)]

    dX = np.zeros_like(Xc)

    for k, offset in enumerate(offsets):
        slices[axis] = ind + offset
        dX += coeffs[k] * X[tuple(slices)]

    return Xc, dX, ind

def ctr_FD(f, h, order):
    N = len(f)
    df = np.empty_like(f)

    x_vals = list(range(order+1))
    
    if order == 8:
        # Central difference (i = 4 to N-5)
        for i in range(4, N-4):
            df[i] = (3*f[i-4] - 32*f[i-3] + 168*f[i-2] - 672*f[i-1]
                    + 672*f[i+1] - 168*f[i+2] + 32*f[i+3] - 3*f[i+4]) / (840*h)

        scale = 840
        coeffs0 = np.round(finite_diff_coeffs(x_vals, x0=0, derivative_order=1) * scale).astype(int).squeeze()
        coeffs1 = np.round(finite_diff_coeffs(x_vals, x0=1, derivative_order=1) * scale).astype(int).squeeze()
        coeffs2 = np.round(finite_diff_coeffs(x_vals, x0=2, derivative_order=1) * scale).astype(int).squeeze()
        coeffs3 = np.round(finite_diff_coeffs(x_vals, x0=3, derivative_order=1) * scale).astype(int).squeeze()

        # Forward-biased stencils (last 4 points)
        df[0] = f[:9] @ coeffs0
        df[1] = f[:9] @ coeffs1
        df[2] = f[:9] @ coeffs2
        df[3] = f[:9] @ coeffs3

        # Backward-biased stencils (last 4 points)
        df[-4] = f[-9:] @ -coeffs3[::-1]
        df[-3] = f[-9:] @ -coeffs2[::-1]
        df[-2] = f[-9:] @ -coeffs1[::-1]
        df[-1] = f[-9:] @ -coeffs0[::-1]
        
    if order == 10:
        for i in range(5, N-5):
            df[i] = (-2*f[i-5] + 25*f[i-4] - 150*f[i-3] + 600*f[i-2] - 2100*f[i-1]
                    + 2100*f[i+1] - 600*f[i+2] + 150*f[i+3] - 25*f[i+4] + 2*f[i+5]) / (2520*h)
        
        scale = 2520
        coeffs0 = np.round(finite_diff_coeffs(x_vals, x0=0, derivative_order=1) * scale).astype(int).squeeze()
        coeffs1 = np.round(finite_diff_coeffs(x_vals, x0=1, derivative_order=1) * scale).astype(int).squeeze()
        coeffs2 = np.round(finite_diff_coeffs(x_vals, x0=2, derivative_order=1) * scale).astype(int).squeeze()
        coeffs3 = np.round(finite_diff_coeffs(x_vals, x0=3, derivative_order=1) * scale).astype(int).squeeze()
        coeffs4 = np.round(finite_diff_coeffs(x_vals, x0=4, derivative_order=1) * scale).astype(int).squeeze()

        # Forward-biased stencils (last 5 points)
        df[0] = f[:11] @ coeffs0
        df[1] = f[:11] @ coeffs1
        df[2] = f[:11] @ coeffs2
        df[3] = f[:11] @ coeffs3
        df[4] = f[:11] @ coeffs4

        # Backward-biased stencils (last 5 points)
        df[-5] = f[-11:] @ -coeffs4[::-1]
        df[-4] = f[-11:] @ -coeffs3[::-1]
        df[-3] = f[-11:] @ -coeffs2[::-1]
        df[-2] = f[-11:] @ -coeffs1[::-1]
        df[-1] = f[-11:] @ -coeffs0[::-1]

    if order == 12:
        for i in range(6, N-6):
            df[i] = (5*f[i-6] - 72*f[i-5] + 495*f[i-4] - 2200*f[i-3] + 7425*f[i-2] - 23760*f[i-1]
                    + 23760*f[i+1] - 7425*f[i+2] + 2200*f[i+3] - 495*f[i+4] + 72*f[i+5] - 5*f[i+6]) / (27720*h)
        
        scale = 27720
        coeffs0 = np.round(finite_diff_coeffs(x_vals, x0=0, derivative_order=1) * scale).astype(int).squeeze()
        coeffs1 = np.round(finite_diff_coeffs(x_vals, x0=1, derivative_order=1) * scale).astype(int).squeeze()
        coeffs2 = np.round(finite_diff_coeffs(x_vals, x0=2, derivative_order=1) * scale).astype(int).squeeze()
        coeffs3 = np.round(finite_diff_coeffs(x_vals, x0=3, derivative_order=1) * scale).astype(int).squeeze()
        coeffs4 = np.round(finite_diff_coeffs(x_vals, x0=4, derivative_order=1) * scale).astype(int).squeeze()
        coeffs5 = np.round(finite_diff_coeffs(x_vals, x0=5, derivative_order=1) * scale).astype(int).squeeze()

        # Forward-biased stencils (last 5 points)
        df[0] = f[:13] @ coeffs0
        df[1] = f[:13] @ coeffs1
        df[2] = f[:13] @ coeffs2
        df[3] = f[:13] @ coeffs3
        df[4] = f[:13] @ coeffs4
        df[5] = f[:13] @ coeffs5

        # Backward-biased stencils (last 5 points)
        df[-6] = f[-13:] @ -coeffs5[::-1]
        df[-5] = f[-13:] @ -coeffs4[::-1]
        df[-4] = f[-13:] @ -coeffs3[::-1]
        df[-3] = f[-13:] @ -coeffs2[::-1]
        df[-2] = f[-13:] @ -coeffs1[::-1]
        df[-1] = f[-13:] @ -coeffs0[::-1]
        
    return df