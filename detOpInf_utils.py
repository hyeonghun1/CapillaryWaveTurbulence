import numpy as np
import scipy.linalg as la
import opinf
from scipy.integrate import solve_ivp

class OpInfROM:
    def __init__(self, V, Q, t, sys_type="quadratic"):
        self.V = V
        self.Q = Q
        self.t = t
        self.dt = np.mean(np.diff(t))
        self.sys_type = sys_type
        
        self.r = None
        self.Vr = None
        self.Q_hat = None
        
        self.D = None
        self.R = None
        
        self.c_hat = None
        self.A_hat = None
        self.H_hat = None

    def compress_data(self, r):
        self.r = r
        self.Vr = self.V[:, :r]
        Q_hat = self.Vr.T @ self.Q
        return Q_hat

    def decompress_data(self, Q_hat):
        Q_recon = self.Vr @ Q_hat
        return Q_recon

    def regularizer(self, lambda1, lambda2=None):
        """
        Get regularizer diagonal matrix for linear or quadratic systems.
        
        Parameters:
            r (int): reduced dimension
            lambda1 (float): linear regularizer
            lambda2 (float, optional): quadratic regularizer (if None, assumes linear system)
        Returns:
            Reg (ndarray): regularizer
        """
        if lambda2 is None:  # Linear case
            d = 1 + self.r
            Reg = np.ones(d) * lambda1
        else:                # Quadratic case
            r1 = 1 + self.r
            r2 = self.r * (self.r + 1) // 2
            d = r1 + r2
            Reg = np.zeros(d)
            Reg[:r1] = lambda1  # Regularizer for A
            Reg[r1:] = lambda2  # Regularizer for H
        
        return np.diag(Reg)

    def data_mat_generator(self, q_hat):
        
        """Generate data matrix D
        Args:
            q_hat (ndarray): reduced states (r,k)
        Returns:
            D (ndarray): data matrix with constant, linear (and quadratic) data from reduced states
                ( k, d(r) ) -> d(r) = 1 + r + r*(r+1)/2
        """
        
        k = q_hat.shape[1]
        
        if self.sys_type == "linear":
            D = np.hstack((np.ones(k).reshape(-1,1), q_hat.T))
        elif self.sys_type == "quadratic":
            QkronQ = opinf.operators.QuadraticOperator.ckron(q_hat)
            D = np.hstack((np.ones(k).reshape(-1,1), q_hat.T, QkronQ.T))
        else:
            raise ValueError("Invalid system type specified.")
            
        return D

    def infer_operator(self, D, R, lambdas):
        """Infer linear / quadratic operators from data.
        Parameters:
            r (int): reduced dimension
            D (ndarray): data matrix
            R (ndarray): d/dt matrix
            sys_type (str): system model type
            lambdas (tupple): regularizers
        Returns:
            c_hat: constant operator
            A_hat: linear operator
            H_hat: quadratic operator
        """
        
        if self.sys_type == "linear":
            Gamma = self.regularizer(lambda1=lambdas)
        elif self.sys_type == "quadratic":
            Gamma = self.regularizer(lambda1=lambdas[0], lambda2=lambdas[1])
        else:
            raise ValueError("Invalid system type specified.")
        
        A_modified_normal = D.T @ D + Gamma.T @ Gamma
        b_modified_normal = D.T @ R.T
        Ot = la.pinv(A_modified_normal) @ b_modified_normal
        O = Ot.T
        
        self.c_hat = O[:, 0]
        self.A_hat = O[:, 1:self.r+1]
        self.H_hat = O[:, self.r+1:] if self.sys_type == "quadratic" else None
        
        return (self.c_hat, self.A_hat) if self.sys_type == "linear" else (self.c_hat, self.A_hat, self.H_hat)

    def ROMfunc(self, t, q):
        """Get right-handside of a reduced dynamical system of equation.
        dydt = chat + Ahat @ q 
                or
        dydt = chat + Ahat @ q + Hhat @ (q x q)

        Paramters:
            t (_type_): time
            q (ndarray): reduced states
            chat (ndarray)
            Ahat (ndarray) 
            Hhat (ndarray)
        Returns:
            dydt (ndarray): right-handside of ROM
        """
        if self.H_hat is None:    # Linear OpInf
            dydt = self.c_hat + self.A_hat @ q        
            
        else:               # Quadratic OpInf
            q_kron_q = opinf.operators.QuadraticOperator.ckron(q)
            dydt = self.c_hat + self.A_hat @ q + self.H_hat @ (q_kron_q) 
            
        return dydt

    def linear_opinf_train_error(self, D, R, train_time, Q_hat, Q_hat_norm, lambdas):
        
        q0_ = Q_hat[:,0]
        c_hat, A_hat = self.infer_operator(D, R, lambdas)
        # args = (c_hat, A_hat, None)
        q_rom = solve_ivp(self.ROMfunc, t_span=(train_time[0], train_time[-1]), y0=q0_, method='RK45', 
                        t_eval=train_time).y
        
        if q_rom.shape[1] == len(train_time):
            error = la.norm(Q_hat - q_rom) / Q_hat_norm
        else:
            error = np.nan

        return error

    def quadratic_opinf_train_error(self, D, R, train_time, Q_hat, Q_hat_norm, lambdas):
        
        q0_ = Q_hat[:,0]
        c_hat, A_hat, H_hat = self.infer_operator(D, R, lambdas)
        # args = (c_hat, A_hat, H_hat)
        q_rom = solve_ivp(self.ROMfunc, t_span=(train_time[0], train_time[-1]), y0=q0_, method='RK45', t_eval=train_time).y
        
        if q_rom.shape[1] != len(train_time):
            error = np.nan
        else:        
            error = la.norm(Q_hat - q_rom) / Q_hat_norm
        
        return error
    
    def grid_search(self, D, R, t, Q_hat):
        
        lambda1 = np.logspace(-5, +10, 10)
        lambda2 = np.logspace(0, +12, 10)
        lambda1_grid, _ = np.meshgrid(lambda1, lambda2)
        error_quad = np.zeros_like(lambda1_grid)

        q_hat_norm = la.norm(Q_hat)
        
        for j, reg2 in enumerate(lambda2):
            for i, reg1 in enumerate(lambda1):
                try:
                    err = self.quadratic_opinf_train_error(D, R, t, Q_hat, q_hat_norm, [reg1, reg2])
                    error_quad[j,i] = err         # Save only if no error occurs
                except la.LinAlgError:
                    # print(f"Skipping loop for: {i}, {j}")
                    error_quad[j,i] = np.nan      # save Nan for errors
                    continue
                except ValueError:
                    # print(f"Skipping loop for: {i}, {j}")
                    error_quad[j, i] = np.nan
                    continue

        min_idx = np.nanargmin(error_quad)
        min_index_2d = np.unravel_index(min_idx, error_quad.shape)
        # quad_reg_optimal = [lambda1[min_index_2d[1]], lambda2[min_index_2d[0]]]
        min_error_quad = error_quad[min_index_2d]
        
        return min_error_quad, [min_index_2d[1], min_index_2d[0]]

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