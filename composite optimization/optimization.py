from collections import defaultdict
import numpy as np
import time
import oracles
import scipy
from oracles import lasso_duality_gap

def check_error(obj):
    if np.isnan(obj).any() or np.isinf(obj).any():
        if display:
            print("Computational error!")
        return obj, 'computational_error'
    return False

def subgradient_method(oracle, x_0, tolerance=1e-2, max_iter=1000, alpha_0=1,
                       display=False, trace=False):
    """
    Subgradient descent method for nonsmooth convex optimization.
    Parameters
    ----------
    oracle : BaseNonsmoothConvexOracle-descendant object
        Oracle with .func() and .subgrad() methods implemented for computing
        function value and its one (arbitrary) subgradient respectively.
        If available, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    alpha_0 : float
        Initial value for the sequence of step-sizes.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    Returns
    -------
    best_x : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values phi(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    
    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0)
    start = time.time()
    best_x = np.copy(x_0)
    best_func = oracle.func(x_k)

    def make_history():
        if display:
            print("Optimization is going")
        if not trace:
            return
        history['time'].append(time.time() - start)
        history['func'].append(f_k)
        history['duality_gap'].append(gap)
        if x_k.shape[0] < 3:
            history['x'].append(np.copy(x_k))

    for k in range(max_iter+1):
        f_k = oracle.func(x_k)
        phi = oracle.subgrad(x_k)
        gap = oracle.duality_gap(x_k)
        
        make_history()
        best_x = x_k if f_k < best_func else best_x
        best_func = min(best_func, f_k)
            
        if gap <= tolerance:
            return best_x, 'success', history
            
        x_k = x_k - alpha_0 * phi / np.linalg.norm(phi) / (k + 1) ** 0.5
        
    return best_x, 'iterations_exceeded', history


def proximal_gradient_method(oracle, x_0, L_0=1.0, tolerance=1e-5,
                             max_iter=1000, trace=False, display=False, need_iters=False):
    """
    Gradient method for composite optimization.
    Parameters
    ----------
    oracle : BaseCompositeOracle-descendant object
        Oracle with .func() and .grad() and .prox() methods implemented
        for computing function value, its gradient and proximal mapping
        respectively.
        If available, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    L_0 : float
        Initial value for adaptive line-search.
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    Returns
    -------
    best_x : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of objective function values phi(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0)
    start = time.time()
    best_x = np.copy(x_0)
    best_phi = oracle.func(x_k)
    iters = 0
    
    def make_history():
        if display:
            print("Optimization is going")
        if not trace:
            return
        history['time'].append(time.time() - start)
        history['func'].append(phi_k)
        history['duality_gap'].append(gap)
        if x_k.shape[0] < 3:
            history['x'].append(np.copy(x_k))
        if need_iters:
            history['iters'].append(iters)
 
    L = L_0
    for k in range(max_iter+1):
        phi_k = oracle.func(x_k)
        f_k = oracle._f.func(x_k)
        gap = oracle.duality_gap(x_k)
        
        make_history()

        best_x = x_k if f_k < best_phi else best_x
        best_phi = min(best_phi, phi_k)
            
        if gap <= tolerance:
            return best_x, 'success', history

        grad_f_k = oracle._f.grad(x_k)

        while True:
            iters += 1
            y = oracle.prox(x_k - grad_f_k / L, 1/L)
            if oracle._f.func(y) <= f_k + grad_f_k.dot(y - x_k) + L / 2 * np.linalg.norm(y - x_k) ** 2:
                break
            L *= 2

        x_k = y
        L = max(L_0, L/2)
        
    return best_x, 'iterations_exceeded', history

def armijo(oracle, start, x, d, c1):
    a = start
    f = oracle.func(x)
    df = oracle.grad_directional(x, d, 0)
    while oracle.func_directional(x, d, a) > f + c1 * a * df:
      a /= 2
    return a

def alpha(x, d):
    theta=0.99
    ans = [1.]
    x, u = np.split(x, 2)
    d_x, d_u = np.split(d, 2)

    for i in range(len(d_x)):
      if d_x[i] > d_u[i]:
        ans.append(theta * (u[i]-x[i]) / (d_x[i]-d_u[i]))
      if d_x[i] < -d_u[i]:
        ans.append(-theta * (u[i]+x[i]) / (d_x[i]+d_u[i]))
    return min(ans)

def newton(c1, oracle, x_0, max_iter, tolerance=1e-5, trace=False, display=False):
    x_k = np.copy(x_0)
    grad = oracle.grad(x_k)
    check_error(grad)
    stop = tolerance * grad.dot(grad)

    for _ in range(max_iter+1):
        grad = oracle.grad(x_k)
        check_error(grad)
        check_error(grad.dot(grad))

        if grad.dot(grad) <= stop:
            return x_k, 'success'
        
        try:
            d_k = scipy.linalg.cho_solve(scipy.linalg.cho_factor(oracle.hess(x_k)), -grad)
        except scipy.linalg.LinAlgError:
            return x_k, 'computational_error'
        
        a_k = armijo(oracle, alpha(x_k, d_k), x_k, d_k, c1)
        check_error(a_k)
        x_k = x_k + a_k * d_k
        check_error(x_k)

    return x_k, 'iterations_exceeded'

def barrier_method_lasso(A, b, reg_coef, x_0, u_0, tolerance=1e-5,
                         tolerance_inner=1e-8, max_iter=100,
                         max_iter_inner=20, t_0=1, gamma=10,
                         c1=1e-4, lasso_duality_gap=None,
                         trace=False, display=False):
    """
    Log-barrier method for solving the problem:
        minimize    f(x, u) := 1/2 * ||Ax - b||_2^2 + reg_coef * \sum_i u_i
        subject to  -u_i <= x_i <= u_i.
    The method constructs the following barrier-approximation of the problem:
        phi_t(x, u) := t * f(x, u) - sum_i( log(u_i + x_i) + log(u_i - x_i) )
    and minimize it as unconstrained problem by Newton's method.
    In the outer loop `t` is increased and we have a sequence of approximations
        { phi_t(x, u) } and solutions { (x_t, u_t)^{*} } which converges in `t`
    to the solution of the original problem.
    Parameters
    ----------
    A : np.array
        Feature matrix for the regression problem.
    b : np.array
        Given vector of responses.
    reg_coef : float
        Regularization coefficient.
    x_0 : np.array
        Starting value for x in optimization algorithm.
    u_0 : np.array
        Starting value for u in optimization algorithm.
    tolerance : float
        Epsilon value for the outer loop stopping criterion:
        Stop the outer loop (which iterates over `k`) when
            `duality_gap(x_k) <= tolerance`
    tolerance_inner : float
        Epsilon value for the inner loop stopping criterion.
        Stop the inner loop (which iterates over `l`) when
            `|| \nabla phi_t(x_k^l) ||_2^2 <= tolerance_inner * \| \nabla \phi_t(x_k) \|_2^2 `
    max_iter : int
        Maximum number of iterations for interior point method.
    max_iter_inner : int
        Maximum number of iterations for inner Newton's method.
    t_0 : float
        Starting value for `t`.
    gamma : float
        Multiplier for changing `t` during the iterations:
        t_{k + 1} = gamma * t_k.
    c1 : float
        Armijo's constant for line search in Newton's method.
    lasso_duality_gap : callable object or None.
        If calable the signature is lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef)
        Returns duality gap value for esimating the progress of method.
    trace : bool
        If True, the progress information is appended into history dictionary
        during training. Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    Returns
    -------
    (x_star, u_star) : tuple of np.array
        The point found by the optimization procedure.
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every **outer** iteration of the algorithm
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    oracle = oracles.BarrierLassoOracle(A, b, t_0, reg_coef)
    x_k = np.copy(x_0)
    u_k = np.copy(u_0)
    cur = np.concatenate((x_k, u_k))
    start = time.time()

    def process():
        if display:
           print("Optimization is going")
        if not trace:
           return
        history['time'].append(time.time() - start)
        history['func'].append(oracle.func(cur))
        history['duality_gap'].append(duality_gap)
        if len(x_k) < 3:
          history['x'].append(np.copy(x_k))


    for _ in range(max_iter+1):
        x_k, u_k = np.split(cur, 2)
        duality_gap = lasso_duality_gap(x_k, A.dot(x_k) - b, A.T.dot((A.dot(x_k) - b)), b, reg_coef)
        process()
        if duality_gap <= tolerance:
            return (x_k, u_k), 'success', history
        cur, message = newton(c1, oracle, cur, max_iter_inner, tolerance=tolerance_inner)
        oracle.t *= gamma
        if message == 'computational_error':
            return cur, 'computational_error', history

    return cur, 'iterations_exceeded', history