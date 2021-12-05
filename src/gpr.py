from functools import partial
import jax
from jax.config import config
import jax.numpy as np
import jax.random as random
import jax.scipy as scipy
import matplotlib.pyplot as plt
from jax.flatten_util import ravel_pytree
import numpy as onp
import scipy.optimize as opt


def cov_map(cov_func, xs, xs2=None):
    """Compute a covariance matrix from a covariance function and data points.
    Args:
      cov_func: callable function, maps pairs of data points to scalars.
      xs: array of data points, stacked along the leading dimension.
    Returns:
      A 2d array `a` such that `a[i, j] = cov_func(xs[i], xs[j])`.
    """
    if xs2 is None:
        return jax.vmap(lambda x: jax.vmap(lambda y: cov_func(x, y))(xs))(xs)
    else:
        return jax.vmap(lambda x: jax.vmap(lambda y: cov_func(x, y))(xs))(xs2).T



# Note, writing out the vectorized form of the identity
# ||x-y||^2 = <x-y,x-y> = ||x||^2 + ||y||^2 - 2<x,y>
# for computing squared distances would be more efficient (but less succinct).
def exp_quadratic(x1, x2):
    return np.exp(-np.sum((x1 - x2)**2))


# def activation(x):
#     return np.logaddexp(x, 0.)

# def activation(x):
#     return np.exp(x)


def activation(x):
    return x


def gp(params, x, y, xtest=None, compute_marginal_likelihood=False):
    numpts = len(x)
    amp = activation(params['amplitude'])
    ls =  activation(params['lengthscale'])
    noise =  activation(params['noise'])

    x = x / ls
    train_cov = amp*cov_map(exp_quadratic, x) + np.eye(numpts) * (noise)
    chol = scipy.linalg.cholesky(train_cov, lower=True)

    kinvy = scipy.linalg.solve_triangular(chol.T, scipy.linalg.solve_triangular(chol, y, lower=True))
    if compute_marginal_likelihood:
        log2pi = np.log(2. * np.pi)
        ml = np.sum(-0.5 * np.dot(y.T, kinvy) - np.sum(np.log(np.diag(chol))) - (numpts / 2.) * log2pi)
        ml -= np.sum(-0.5 * np.log(2 * np.pi) - np.log(amp)**2) # lognormal prior
        return -ml

    xtest = xtest / ls
    cross_cov = amp*cov_map(exp_quadratic, x, xtest)
    mu = np.dot(cross_cov.T, kinvy)
    v = scipy.linalg.solve_triangular(chol, cross_cov, lower=True)
    var = amp * cov_map(exp_quadratic, xtest) - np.dot(v.T, v)
    return mu, var


marginal_likelihood = partial(gp, compute_marginal_likelihood=True)
predict = jax.jit(partial(gp, compute_marginal_likelihood=False))
grad_fun = jax.jit(jax.grad(marginal_likelihood))


def train_scipy(params, x, y):
    params_ini, unravel = ravel_pytree(params)

    obj_vals = []
    def objective(params):
        params = unravel(params)
        print(f"\n######################### Evaluating objective value - step {objective.counter}")
        objective.counter += 1
        obj_val = marginal_likelihood(params, x, y)
        obj_vals.append(obj_val)
        print(f"obj_val = {obj_val}")
        return obj_val

    def derivative(params):
        params = unravel(params)
        der_val = grad_fun(params, x, y)
        der_val, _ = ravel_pytree(der_val)
        return onp.array(der_val, order='F', dtype=onp.float64)

    bounds = ((1e-3, 10.), (0.01, 10.), (3*1e-4, 3*1e-4))
    # bounds = ((-10., 1.), (-10., 1.), (-10., -1.))
    objective.counter = 0
    options = {'maxiter': 1000, 'disp': True}  # CG or L-BFGS-B or Newton-CG or SLSQP or trust-constr
    res = opt.minimize(fun=objective,
                       x0=params_ini,
                       method='SLSQP',
                       jac=derivative,
                       bounds=bounds,
                       callback=None,
                       options=options)
    print(f"res.x = {res.x}")
    params = unravel(activation(res.x))
    return params


def example():
    # Covariance hyperparameters to be learned
    params = {"amplitude": 0.1,
              "lengthscale": 0.1,
              "noise": 1e-6}   

    # Create a really simple toy 1D function
    numpts = 7
    key = random.PRNGKey(0)
    y_fun = lambda x: np.sin(x) + 0.1 * random.normal(key, shape=(x.shape[0], 1))
    x = (random.uniform(key, shape=(numpts, 1)) * 4.) + 1
    y = y_fun(x)
    xtest = np.linspace(0, 6., 200)[:, None]
    params = train_scipy(params, x, y)
    print(params)
    mu, var = predict(params, x, y, xtest)
    std = np.sqrt(np.diag(var))
    plt.plot(x, y, "k.")
    plt.plot(xtest, mu)
    plt.fill_between(xtest.flatten(), mu.flatten() - std * 2, mu.flatten() + std * 2)
    plt.show()


if __name__ == "__main__":
    example()
