# %%

"""
Accurate gravitational modeling and magnetic field modeling of the earth requires us to be able to compute
spherical harmonics, given modelled coefficents.
"""
from functools import partial
import jumpy as jp
import jax


@partial(jax.jit, static_argnames=['degree'])
def compute_spherical_harmonics(coeffs: jp.ndarray, theta: jp.ndarray, degree: int):
    """
    Here we compute:
    v(c, theta) = sum_n=0^degree sum_m=0^n (c_m * P^m_n(theta))
    """
    lpmn = jax.scipy.special.lpmn_values(z=jp.array([theta]), m=degree, n=degree, is_normalized=True)
    a = coeffs * jax.numpy.moveaxis(lpmn, -1, 0)
    return jp.sum(a)



# %%
