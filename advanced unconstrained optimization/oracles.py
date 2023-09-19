import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')

    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))

    def hess_vec(self, x, v):
        """
        Computes matrix-vector product with Hessian matrix f''(x) v
        """
        return self.hess(x).dot(v)

    def minimize_directional(self, x_k, d_k):
        raise NotImplementedError('minimize_directional is not implemented.')


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A

    def minimize_directional(self, x_k, d_k):
        return -self.grad(x_k).dot(d_k) / self.A.dot(d_k).dot(d_k)


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATy : function of y
            Computes matrix-vector product A^Ty, where y is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        return np.mean(np.logaddexp(0, -1 * self.b * self.matvec_Ax(x))) + self.regcoef * (x @ x) / 2

    def grad(self, x):
        return -1 * self.matvec_ATx(self.b * expit(-self.b * self.matvec_Ax(x))) / len(self.b) + self.regcoef * x

    def hess(self, x):
        t = expit(-self.b * self.matvec_Ax(x))
        return self.matmat_ATsA(t * (1.0 - t)) / len(self.b) + self.regcoef * np.eye(len(x))
 
    def hess_vec(self, x, v):
        t = expit(self.b * self.matvec_Ax(x))
        return self.matvec_ATx(t * (1.0 - t) * self.matvec_Ax(v)) / len(self.b) + self.regcoef * v


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle for logistic regression with l2 regularization
    with optimized *_directional methods (are used in line_search).

    For explanation see LogRegL2Oracle.
    """

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.last_x = None
        self.last_d = None
        self.last_a = None
        self.Ax = None
        self.Ad = None
        self.x_d = None
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)

    def process_Ax(self, x):
        x_copy = np.copy(x)
        if self.last_x is None:
            self.Ax = self.matvec_Ax(x_copy)
        elif np.all(self.x_d == x_copy):
            self.Ax = self.Ax + self.last_a * self.Ad
        elif np.any(self.last_x != x_copy):
            self.Ax = self.matvec_Ax(x_copy)
        self.last_x = x_copy
        return self.Ax

    def process_Ad(self, d):
        d_copy = np.copy(d)
        if self.x_d is None or np.any(self.last_d != d):
            self.Ad = self.matvec_Ax(d_copy)
        self.last_d = d_copy
        return self.Ad

    def func(self, x):
        ans = np.mean(np.logaddexp(0, -1 * self.b * self.process_Ax(x))) + self.regcoef * (x @ x) / 2         
        self.x_d = None
        return ans

    def grad(self, x):
        ans = -1 * self.matvec_ATx(self.b * expit(-self.b * self.process_Ax(x))) / len(self.b) + self.regcoef * x
        self.x_d = None
        return ans

    def hess(self, x):
        t = expit(-self.b * self.process_Ax(x))
        ans = self.matmat_ATsA(t * (1.0 - t)) / len(self.b) + self.regcoef * np.eye(len(x))
        self.x_d = None
        return ans

    def hess_vec(self, x, v):
        t = expit(self.b * self.process_Ax(x))
        ans = (self.matvec_Ax(t * (1 - t) * self.process_Ax(v))) / len(self.b) + self.regcoef * v
        self.x_d = None
        return ans


    def func_directional(self, x, d, alpha):
        foo = self.process_Ax(x) + alpha * self.process_Ad(d)
        cur_x_d = x + d * alpha
        ans = np.mean(np.logaddexp(0, -1 * self.b * foo)) + self.regcoef * (cur_x_d @ cur_x_d) / 2
        self.x_d = cur_x_d
        self.last_a = alpha
        return np.squeeze(ans)

    def grad_directional(self, x, d, alpha):
        Ad = self.process_Ad(d)
        foo = self.process_Ax(x) + alpha * Ad
        cur_x_d = x + d * alpha
        ans = self.regcoef * cur_x_d.dot(d)
        ans = ans - ((self.b * expit(-self.b * foo)).dot(Ad)) / len(self.b)
        self.last_a = alpha
        self.x_d = cur_x_d
        return np.squeeze(ans)


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    def matvec_Ax(x):
        return A.dot(x)

    def matvec_ATx(x):
        return A.T.dot(x)

    def matmat_ATsA(s):
        return A.T.dot(scipy.sparse.diags(s).dot(A))
        #return A.T.dot(A * s.reshape(-1, 1))

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise 'Unknown oracle_type=%s' % oracle_type
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def hess_vec_finite_diff(func, x, v, eps=1e-5):
    """
    Returns approximation of the matrix product 'Hessian times vector'
    using finite differences.
    """

    e = eps * np.eye(len(x))
    hess_vec = []

    for e_i in e:
        hess_vec.append((func(x + v * eps + e_i) - func(x + e_i) - func(x + v * eps) + func(x)) / eps ** 2)
    
    return hess_vec