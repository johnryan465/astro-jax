# %%
from functools import partial

import numpy as np
import pandas as pd

import jax
import jumpy as jp

from astrojax.physics.earth.spherical_harmonics import compute_spherical_harmonics

degree = 13

df = pd.read_csv('igrf13coeffs.txt', sep="\s+", skiprows=3)
df.rename(columns={'2020-25': '2025'}, inplace=True)
# %%
g, h = {}, {}
for year in df.columns[3:]:
    g_arr = np.zeros((degree+1, degree+1))
    _g = df[df["g/h"] == "g"][["n", "m", year]]
    g_arr[_g["n"].values, _g["m"].values] = _g[year].values
    g[int(year[:4])] = g_arr
    h_arr = np.zeros((degree+1, degree+1))
    _h = df[df["g/h"] == "h"][["n", "m", year]]
    h_arr[_h["n"].values, _h["m"].values] = _h[year].values
    h[int(year[:4])] = h_arr

# %%

g_values = jp.array(list(g.values()))
h_values = jp.array(list(h.values()))
keys = jp.array(list(g.keys()))

#@partial(jax.jit, static_argnames=['keys', 'g_values', 'h_values'])
@jax.jit
def interpolate_coeffs(years: jax.numpy.ndarray, keys: jax.numpy.ndarray, g_values: jax.numpy.ndarray, h_values: jax.numpy.ndarray):
    @jax.jit
    def int_fn(years, fp):
        a = jax.numpy.interp(x=years, xp=keys, fp=fp)
        return a

    int_g = jax.vmap(jax.vmap(int_fn, in_axes=(None, 1), out_axes=-1), in_axes=(None, 2), out_axes=-1)(years, g_values)
    int_h = jax.vmap(jax.vmap(int_fn, in_axes=(None, 1), out_axes=-1), in_axes=(None, 2), out_axes=-1)(years, h_values)
    return int_g, int_h



# %%
@partial(jax.jit, static_argnames=['degree'])
def igrf_coeffs_for_spherical_harmonics(a: jp.ndarray, r: jp.ndarray, phi: jp.ndarray, g: jp.ndarray, h: jp.ndarray, degree: int):
    """
    This function returns the coefficients for the spherical harmonics.

    Args:
        a: [...]
        r: [...]
        phi: [...]
        g: [..., degree+1, degree+1]
        h: [..., degree+1, degree+1]


    C_n_m = a (a/r)^n+1 (g_n^m cos (m psi) + h_n^m sin (m psi))
    """
    n = jp.arange(1, degree + 2)
    m = jp.arange(0, degree + 1)
    # pow_fn = jax.tree_util.Partial(jax.numpy.power, x2=n)

    first = jax.numpy.power(a / r, n)

    # prod_fn = jax.tree_util.Partial(jax.numpy.multiply, x2=m)
    m_psi = phi * m
    cos_psi = jp.cos(m_psi)
    sin_psi = jp.sin(m_psi)

    a_g_n_m_cos = jp.outer(first, cos_psi) * g
    a_h_n_m_sin = jp.outer(first, sin_psi) * h
    return jp.multiply( jax.numpy.expand_dims(a, (-1, )), a_g_n_m_cos + a_h_n_m_sin)




# @partial(jax.jit, static_argnames=['keys', 'g_values', 'h_values'])
def create_igrf_value(keys: jp.ndarray, g_values: jp.ndarray, h_values: jp.ndarray):
    @jax.jit
    def igrf_value(r: jp.ndarray, theta: jp.ndarray, phi: jp.ndarray, t: jp.ndarray):
        """
        This function returns the value of the spherical harmonics.

        Args:
            r: [...]
            phi: [...]
            theta: [...]
            t: [...]
            degree: int

        Returns:
            [..., degree+1, degree+1]
        """
        degree = 13
        a = 6731.2 * jp.ones_like(r)
        g, h = interpolate_coeffs(years=t, keys=keys, g_values=g_values, h_values=h_values)
        res = igrf_coeffs_for_spherical_harmonics(a=a, r=r, phi=phi, g=g, h=h, degree=degree)
        return compute_spherical_harmonics(degree=degree, coeffs=res, theta=jp.cos(theta))
    return igrf_value
    


# %%
years= 2010.0 * jp.ones((10,)).astype(jp.float32)
theta = (jp.pi / 2)* jp.ones((10,)).astype(jp.float32)
r = jp.ones((10,)) * 6731.2
phi = jp.zeros((10,)).astype(jp.float32)


# %%
igrf_value = create_igrf_value(keys=keys, g_values=g_values, h_values=h_values)
jax.vmap(igrf_value)(r, theta, phi, years)

# %%
j = jax.jit(jax.grad(igrf_value, argnums=(0, 1, 2))) # (r, theta, phi, years)

compiled_grad = jax.vmap(j)
compiled_grad(r, theta, phi, years)


# %%

# %%

# %%

# %%

# %%
