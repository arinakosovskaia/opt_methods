from collections import defaultdict, deque  # Use this for effective implementation of L-BFGS
import scipy
import time
import numpy as np
from scipy.linalg import cho_solve, cho_factor, LinAlgError
from utils import get_line_search_tool


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradient descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options={'method': 'Armijo', 'c1': 1e-4})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    start = time.time()
    gradient_x_0 = oracle.grad(x_0)
    criterion = tolerance * (np.linalg.norm(gradient_x_0) ** 2)

    def check(a):
      if np.isnan(a).any() or np.isinf(a).any():
        if display:
          print("optimization is ended")
        return False
      return True

    if not check(gradient_x_0):
      return x_k, "computational_error", history

    if display:
      print("optimization is started")

    for iter in range(max_iter+1):
        derivative_x_k = oracle.grad(x_k)
        derivative_x_k_norm = (np.linalg.norm(derivative_x_k)) ** 2

        if not (check(derivative_x_k) and check(x_k) and check(derivative_x_k_norm)):
          return x_k, "computational_error", history

        derivative_x_k *= -1

        if trace:
          history['time'].append(time.time() - start)
          func = oracle.func(x_k)
          norm_sqrt = np.sqrt(derivative_x_k_norm)
          if not (check(func) and check(norm_sqrt)):
            return x_k, "computational_error", history
          history['func'].append(oracle.func(x_k))
          history['grad_norm'].append(norm_sqrt)
          if x_k.shape[0] < 3:
            history['x'].append(np.copy(x_k))

        if derivative_x_k_norm <= criterion:
          if display:
            print("optimization is ended")
          return x_k, 'success', history

        a = line_search_tool.line_search(oracle, x_k, derivative_x_k)
        x_k = x_k + (a * derivative_x_k)

        if not (check(a) and check(x_k)):
          return x_k, "computational_error", history

        if display:
          print("next step")

    if display:
      print("optimization is ended")

    if derivative_x_k_norm > criterion:
      return x_k, "iterations_exceeded", history
    
    return x_k, 'success', history

def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    """
    Newton's optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively. If the Hessian
        returned by the oracle is not positive-definite method stops with message="newton_direction_error"
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'newton_direction_error': in case of failure of solving linear system with Hessian matrix (e.g. non-invertible matrix).
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = newton(oracle, np.zeros(5), line_search_options={'method': 'Constant', 'c': 1.0})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    gradient_x_0 = oracle.grad(x_0)
    criterion = tolerance * (np.linalg.norm(gradient_x_0) ** 2)

    if display:
      print("optimization is started")

    def check(a):
      if np.isnan(a).any() or np.isinf(a).any():
        return False
      return True

    start = time.time()

    for iter in range(max_iter+1):
        derivative_x_k = oracle.grad(x_k)
        derivative_x_k_norm = (np.linalg.norm(derivative_x_k)) ** 2
        if not (check(derivative_x_k) and check(derivative_x_k_norm)):
          return x_k, "computational_error", history
        
        derivative_x_k *= -1

        if trace:
          history['time'].append(time.time() - start)
          func = oracle.func(x_k)
          history['func'].append(func)
          history['grad_norm'].append(np.sqrt(derivative_x_k_norm))
          if not (check(func) and check(np.sqrt(derivative_x_k_norm))):
            return x_k, "computational_error", history
          if x_k.shape[0] < 3:
            history['x'].append(np.copy(x_k))

        if derivative_x_k_norm <= criterion:
          if display:
            print("optimization is ended")
          return x_k, 'success', history

        try:
            hess = oracle.hess(x_k)
            a, b = cho_factor(hess)
            d_k = cho_solve((a, b), derivative_x_k)
        except LinAlgError:
          if not np.any(hess) or not np.any(a): 
              return x_k, "computational_error", history
          if not (check(hess) and check(a) and check(b) and check(d_k)): 
              return x_k, "computational_error", history
          return x_k, 'newton_direction_error', history

        if iter == max_iter:
          break

        alpha = line_search_tool.line_search(oracle, x_k, d_k)
        x_k += alpha * d_k
        if not (check(alpha) and check(x_k)):
          return x_k, "computational_error", history

    if display:
      print("optimization is ended")

    if derivative_x_k_norm > criterion:
      return x_k, "iterations_exceeded", history

    return x_k, 'success', history
