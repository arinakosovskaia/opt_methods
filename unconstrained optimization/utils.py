import numpy as np
import scipy
import scipy.optimize

class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
            - 'Best' -- optimal step size inferred via analytical minimization.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        elif self._method == 'Best':
            pass
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, dict):
        return cls(**dict)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
      if self._method == 'Constant':
        return self.c

      if previous_alpha is None:
        previous_alpha = self.alpha_0

      def phi(a):
        return oracle.func_directional(x_k, d_k, a)

      def derivative_phi(a):
        return oracle.grad_directional(x_k, d_k, a)

      def backtracking(a):
        phi_0 = phi(0)
        derivative_phi_0 = derivative_phi(0)
        while phi(a) > phi_0 + self.c1 * a * derivative_phi_0:
          a /= 2
        return a

      if self._method == 'Wolfe':
        alpha = scipy.optimize.line_search(f=oracle.func, myfprime=oracle.grad, xk=x_k, pk=d_k, c1=self.c1, c2=self.c2)[0]

        if alpha:
          return alpha
      
      return backtracking(previous_alpha) #if Armijo or Wolfe returned None

def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()