import numpy as np

def lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef):
    """
    Estimates f(x) - f* via duality gap for 
        f(x) := 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
    """
    den = np.linalg.norm(ATAx_b, ord=np.inf)

    if not den:
        mu = Ax_b
    else:
        mu = np.min([1., regcoef / den]) * Ax_b

    eta = 1/2 * np.linalg.norm(Ax_b) ** 2 + regcoef * np.linalg.norm(x, ord=1) + 1/2 * np.linalg.norm(mu) ** 2 + b.dot(mu)

    return eta

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