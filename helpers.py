import numpy as np
from scipy.sparse import eye as speye
import math

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

    # m = u_train{1}.shape[0]
    # p = numel(E_train);
    # u = u_train{ii};

    # Central finite difference (accuracy 2)
    # Returns Er (states), Er_dot (time derivatives), ind (indices used)
    Er, Er_dot, ind = central_finite_differences(E_train, h, ord=2, axis=1)
    
    
    # # Build regression matrices
    # for jj in range(min(len(ind), s)):
    #     print(jj)
    #     if isbilinear:
    #         # system has bilinear term
    #         # Python equivalent of MATLAB kron(u, Er) would require u
    #         # Here we assume u is defined elsewhere or passed in
    #         D_new = np.hstack([D, np.vstack([Er[:, jj], np.kron(u[:, jj], Er[:, jj])])])
    #     else:
    #         # no bilinear term
    #         # D_new = np.hstack([D, Er[:, jj:jj+1]])
    #         # print(Er[:, jj:jj+1].shape)
            
    #         D_new = np.hstack([D, Er[:, jj+1]])
        
    #     D = D_new
        
    #     print(D.shape)
        
    #     rhs = np.hstack([rhs, Er_dot[:, jj:jj+1]])
    
    D = Er
    rhs = Er_dot

    print(f"cond(D) = {np.linalg.cond(D):.4e}")


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
    Cr, Cr_dot, ind = central_finite_differences(C_train, h, ord=2, axis=2)

    # Loop over valid indices
    for jj in range(len(ind)):
        Psi_hat = Ahat
        Clyap = Psi_hat @ Cr[:, :, jj]

        # Increment for diffusion contribution
        incre = Cr_dot[:, :, jj] - Clyap - Clyap.T
        Hhat += incre / len(ind)   # mean over indices

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

        # Accumulate f1 and f2
        f1 += (Nb / L_test) * f1_temp
        f2 += (Nb / L_test) * f2_temp

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
    f1 = np.mean(np.linalg.norm(X_T_recon, axis=0)**2)

    # f2: mean over elements of X(T)^3 / exp(X(T))
    f2 = np.mean(X_T_recon**3 / np.exp(X_T_recon))

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
