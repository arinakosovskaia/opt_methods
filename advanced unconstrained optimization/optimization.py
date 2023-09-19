from time import time
import warnings
from collections import deque, defaultdict
import numpy as np
from numpy.linalg import norm
import scipy
import scipy.sparse
import scipy.optimize

from utils import get_line_search_tool

def conjugate_gradients(matvec, b, x_0, tolerance=1e-4, max_iter=None, trace=False, display=False, comp=False, prev_tolerance = 1e-5):
    """
    Solves system Ax=b using Conjugate Gradients method.

    Parameters
    ----------
    matvec : function
        Implement matrix-vector product of matrix A and arbitrary vector x
    b : 1-dimensional np.array
        Vector b for the system.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
        Stop optimization procedure and return x_k when:
         ||Ax_k - b||_2 <= tolerance * ||b||_2
    max_iter : int, or None
        Maximum number of iterations. if max_iter=None, set max_iter to n, where n is
        the dimension of the space
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display:  bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['residual_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    start = time()
    history = defaultdict(list) if trace else None
    if max_iter is None:
        max_iter = 2 * len(b)

    x_k = np.copy(x_0)
    g_k = matvec(x_k) - b
    d_k = -1 * g_k

    if comp:
        stop = np.sqrt(prev_tolerance) * np.linalg.norm(matvec(x_0) - b)
    else:
        stop = tolerance * np.linalg.norm(b)
    
    if display:
         print("Optimization has started")

    def make_history():
        if display:
            print("Optimization is going")
        if trace:
            history['time'].append(time() - start)
            history['residual_norm'].append(norm_g_k)
            if x_k.shape[0] < 3:
                history['x'].append(np.copy(x_k))

    for _ in range(max_iter):
        norm_g_k = np.linalg.norm(np.copy(g_k))
        make_history()
        if norm_g_k <= stop:
            if display:
                print("Optimization is ended")
            return x_k, 'success', history
        Ad_k = matvec(d_k)
        a = g_k.dot(g_k) / d_k.dot(Ad_k)
        x_k = x_k + a * d_k
        g_k = g_k + a * Ad_k
        d_k = -g_k + d_k * g_k.dot(g_k) / (norm_g_k) ** 2

    norm_g_k = np.linalg.norm(np.copy(g_k))
    make_history()
    if norm_g_k > stop:
        if display:
            print("Optimization is failed")
        return x_k, 'iterations_exceeded', history

    if display:
        print("Optimization is ended")
    return x_k, 'success', history

def lbfgs(oracle, x_0, tolerance=1e-4, max_iter=500, memory_size=10,
          line_search_options=None, display=False, trace=False):
    """
    Limited-memory Broyden–Fletcher–Goldfarb–Shanno's method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func() and .grad() methods implemented for computing
        function value and its gradient respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    memory_size : int
        The length of directions history in L-BFGS method.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    stop =  tolerance * (np.linalg.norm(oracle.grad(x_0)) ** 2)
    start = time()
    H = deque()

    def make_history():
        if trace:
            history['time'].append(time() - start)
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(grad_x_k_norm ** (0.5))
            if x_k.shape[0] < 3:
                history['x'].append(x_k)
        if display:
            print("Optimization is going")

    def bfgs_multiply(v, H, gamma_0):
        if not len(H):
            return v * gamma_0
        s, y = H.pop()
        v_s = v - (np.dot(s, v) / np.dot(y, s)) * y
        z = bfgs_multiply(v_s, H, gamma_0)
        return z + ((np.dot(s, v) - np.dot(y, z)) / np.dot(y, s)) * s

    def lbfgs_direction():
        if not len(H):
            s, y = 1, 1
        else:
            s, y = H[-1]
        gamma_0 = np.dot(y, s) / np.dot(y, y)
        if not len(H):
            return -grad_x_k * gamma_0
        return bfgs_multiply(-grad_x_k, H.copy(), gamma_0)


    for iter in range(max_iter+1):
        grad_x_k = oracle.grad(x_k)
        grad_x_k_norm = np.linalg.norm(grad_x_k) ** 2
        make_history()

        if grad_x_k_norm <= stop:
            if display:
                print("Optimization is ended")
            return x_k, 'success', history

        if iter == max_iter:
            break

        if iter:
            if len(H) > memory_size and memory_size != 0:
                H.popleft()
            H.append((x_k - x_k_old, grad_x_k - grad_x_k_old))

        grad_x_k_old = np.copy(grad_x_k)
        x_k_old = np.copy(x_k)

        d_k = lbfgs_direction()
        a = line_search_tool.line_search(oracle, x_k, d_k)
        x_k = x_k + a * d_k
        grad_x_k = oracle.grad(x_k)

        if memory_size == 0:
            H = []

    if display:
        print("Optimization is failed")
    return x_k, "iterations_exceeded", history

def hessian_free_newton(oracle, x_0, tolerance=1e-4, max_iter=500, 
                        line_search_options=None, linear_solver_options=None,
                        display=False, trace=False):
    """
    Hessian Free method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess_vec() methods implemented for computing
        function value, its gradient and matrix product of the Hessian times vector respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    linear_solver_options : dict or None
        Dictionary with parameters for newton's system solver.
        NOTE: Specify it by yourself if you need to setup inner CG method.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    start = time()

    if display:
         print("Optimization has started")

    def make_history():
        if display:
            print("Optimization is going")
        if trace:
            history['time'].append(time() - start)
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(grad_x_k_norm ** (1/2))
            if x_k.shape[0] < 3:
                history['x'].append(np.copy(x_k))

    def matvec(d):
        return oracle.hess_vec(x_k, d)

    stop = tolerance * (np.linalg.norm(oracle.grad(x_0)) ** 2)
    for iter in range(max_iter+1):
        grad_x_k = oracle.grad(x_k)
        grad_x_k_norm = (np.linalg.norm(grad_x_k) ** 2)
        make_history()
        if grad_x_k_norm <= stop:
            if display:
                print("Optimization is ended")
            return x_k, 'success', history

        if iter == max_iter:
            break

        nu = min(0.5, grad_x_k_norm ** (1/4))
        d_k = conjugate_gradients(matvec, -grad_x_k, -grad_x_k, tolerance=nu)[0]

        while d_k.dot(grad_x_k) >= 0:
            d_k = conjugate_gradients(matvec, -grad_x_k, -grad_x_k, tolerance=nu)[0]
            nu /= 10

        a = line_search_tool.line_search(oracle, x_k, d_k)
        x_k = x_k + a * d_k

    if display:
        print("Optimization is failed")
    return x_k, 'iterations_exceeded', history

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
    start = time()
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
          history['time'].append(time() - start)
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
