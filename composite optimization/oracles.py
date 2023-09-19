import numpy as np
import scipy
from scipy.special import expit
from numpy import linalg as LA

class BaseSmoothOracle(object):
    """
    Base class for smooth function.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func is not implemented.')

    def grad(self, x):
        """
        Computes the gradient vector at point x.
        """
        raise NotImplementedError('Grad is not implemented.')

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


class BaseProxOracle(object):
    """
    Base class for proximal h(x)-part in a composite function f(x) + h(x).
    """

    def func(self, x):
        """
        Computes the value of h(x).
        """
        raise NotImplementedError('Func is not implemented.')

    def prox(self, x, alpha):
        """
        Implementation of proximal mapping.
        prox_{alpha}(x) := argmin_y { 1/(2*alpha) * ||y - x||_2^2 + h(y) }.
        """
        raise NotImplementedError('Prox is not implemented.')


class BaseCompositeOracle(object):
    """
    Base class for the composite function.
    phi(x) := f(x) + h(x), where f is a smooth part, h is a simple part.
    """

    def __init__(self, f, h):
        self._f = f
        self._h = h

    def func(self, x):
        """
        Computes the f(x) + h(x).
        """
        return self._f.func(x) + self._h.func(x)

    def grad(self, x):
        """
        Computes the gradient of f(x).
        """
        return self._f.grad(x)

    def prox(self, x, alpha):
        """
        Computes the proximal mapping.
        """
        return self._h.prox(x, alpha)

    def duality_gap(self, x):
        """
        Estimates the residual phi(x) - phi* via the dual problem, if any.
        """
        return None


class BaseNonsmoothConvexOracle(object):
    """
    Base class for implementation of oracle for nonsmooth convex function.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func is not implemented.')

    def subgrad(self, x):
        """
        Computes arbitrary subgradient vector at point x.
        """
        raise NotImplementedError('Subgrad is not implemented.')

    def duality_gap(self, x):
        """
        Estimates the residual phi(x) - phi* via the dual problem, if any.
        """
        return None

class LeastSquaresOracle(BaseSmoothOracle):
    """
    Oracle for least-squares regression.
        f(x) = 0.5 ||Ax - b||_2^2
    """
    def __init__(self, matvec_Ax, matvec_ATx, b):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.b = b

    def func(self, x):
        return 0.5 * LA.norm(self.matvec_Ax(x) - self.b) ** 2

    def grad(self, x):
        return self.matvec_ATx(self.matvec_Ax(x) - self.b)

class L1RegOracle(BaseProxOracle):
    """
    Oracle for L1-regularizer.
        h(x) = regcoef * ||x||_1.
    """
    def __init__(self, regcoef=1):
        self.regcoef = regcoef

    def func(self, x):
        return self.regcoef * LA.norm(x, ord=1)

    def prox(self, x, alpha):
        pr = np.zeros(x.size)
        cond1 = np.where(x < -alpha * self.regcoef)
        cond2 = np.where(x > alpha * self.regcoef)
        pr[cond1] = x[cond1] + alpha * self.regcoef
        pr[cond2] = x[cond2] - alpha * self.regcoef        
        return pr

class LassoProxOracle(BaseCompositeOracle):
    """
    Oracle for 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
        f(x) = 0.5 * ||Ax - b||_2^2 is a smooth part,
        h(x) = regcoef * ||x||_1 is a simple part.
    """
    def __init__(self, f, h):
        if not isinstance(f, LeastSquaresOracle):
            raise ValueError('f must be instance of LeastSquaresOracle')
        if not isinstance(h, L1RegOracle):
            raise ValueError('h must be instance of L1RegOracle')
        super().__init__(f, h)

    def duality_gap(self, x):
        Ax_b = self._f.matvec_Ax(x) - self._f.b
        return lasso_duality_gap(x, Ax_b, self._f.matvec_ATx(Ax_b), self._f.b, self._h.regcoef)

class LassoNonsmoothOracle(BaseNonsmoothConvexOracle):
    """
    Oracle for nonsmooth convex function
        0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
    """
    def __init__(self, matvec_Ax, matvec_ATx, b, regcoef):
        self._least_squares_oracle = LeastSquaresOracle(matvec_Ax, matvec_ATx, b)
        self._regcoef = regcoef
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.b = b
        self.regcoef = regcoef
        
    def func(self, x):
        return 0.5 * LA.norm(self.matvec_Ax(x) - self.b) ** 2 + self.regcoef * LA.norm(x, ord=1)
        
    def subgrad(self, x):
        return self.matvec_ATx(self.matvec_Ax(x) - self.b) + np.sign(x) * self.regcoef
    
    def duality_gap(self, x):
        Ax_b = self.matvec_Ax(x) - self.b
        return lasso_duality_gap(x, Ax_b, self.matvec_ATx(Ax_b), self.b, self.regcoef)
        

def lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef):
    """
    Estimates f(x) - f* via duality gap for
        f(x) := 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
    """
    den = np.linalg.norm(ATAx_b, ord=np.inf)

    mu = Ax_b
    if den:
        mu = np.min([1., regcoef / den]) * Ax_b

    eta = 0.5 * np.linalg.norm(Ax_b) ** 2 + regcoef * np.linalg.norm(x, ord=1) + 0.5 * np.linalg.norm(mu) ** 2 + np.dot(b, mu)

    return eta
    
def create_lasso_prox_oracle(A, b, regcoef):
    def matvec_Ax(x):
        return A.dot(x)

    def matvec_ATx(x):
        return A.T.dot(x)

    return LassoProxOracle(LeastSquaresOracle(matvec_Ax, matvec_ATx, b),
                           L1RegOracle(regcoef))


def create_lasso_nonsmooth_oracle(A, b, regcoef):
    def matvec_Ax(x):
        return A.dot(x)

    def matvec_ATx(x):
        return A.T.dot(x)

    return LassoNonsmoothOracle(matvec_Ax, matvec_ATx, b, regcoef)

class BarrierLassoOracle(BaseSmoothOracle):
    def __init__(self, A, b, t, regcoef):
        self.b = b
        self.t = t
        self.A = A
        self.regcoef = regcoef

        def matvec_Ax_b(x):
          return A.dot(x) - b
        self.matvec_Ax_b = matvec_Ax_b

        def matvec_Ax(x):
          return A.dot(x)
        self.matvec_Ax = matvec_Ax

        def matvec_ATx(x):
          return A.T.dot(x)
        self.matvec_ATx = matvec_ATx
       
        def ATA():
          return A.T.dot(A)
        self.ATA = ATA

        def matvec_ATAx_b(x):
          return A.T.dot(A.dot(x) - b)
        self.matvec_ATAx_b = matvec_ATAx_b

        def mat_ATsA(s):
          return A.T.dot(scipy.sparse.diags(s).dot(A))
        self. mat_ATsA = mat_ATsA

    def func(self, vec):
        x, u = np.split(vec, 2)
        return self.t * (1/2 * np.linalg.norm(self.matvec_Ax_b(x)) ** 2 + self.regcoef * np.sum(u)) - np.sum(np.log(u-x) + np.log(u+x))

    def grad(self, vec):
        x, u = np.split(vec, 2)
        a1 = -1. / (u + x)
        a2 = 1. / (u - x)
        df_x = self.t * self.matvec_ATAx_b(x) + (a1 + a2)
        df_u = self.t * self.regcoef * np.ones(len(x)) + (a1 - a2)
        return np.concatenate((df_x, df_u), axis=0)


    def hess(self, vec):
        x, u = np.split(vec, 2)
        a1 = 1. / (u + x) ** 2
        a2 = 1. / (u - x) ** 2
        xu = np.diag(a1 - a2)
        a4 = np.concatenate((self.t * self.ATA() + np.diag(a1 + a2), xu), axis=0)
        a3 = np.concatenate((xu, np.diag(a1 + a2)), axis=0)
        return np.concatenate((a4, a3), axis=1)