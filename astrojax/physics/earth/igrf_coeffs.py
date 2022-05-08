# %%
from __future__ import division, print_function
from functools import partial
from turtle import right
from typing import Tuple
import jumpy as jp
import jax
from jax.scipy.special import lpmn as _lpmn
from jax.config import config

# config.update('jax_disable_jit', True)
# config.update("jax_enable_x64", True)

a_igrf = 6371.2 #IGRF Earth's radius

@partial(jax.jit, static_argnames=['m', 'n'])
def lpmn(m: int, n: int, z: jp.ndarray) -> Tuple[jp.ndarray, jp.ndarray]:
    return _lpmn(m, n, z) #jax.vmap(_lpmn, in_axes=(None, None, 0))(m, n, z)

@jax.jit
def factorial(n):
    return jax.lax.exp(jax.lax.lgamma(n + 1.0))

@jax.jit
def interpolate(x, xp, fp):
    in_fn = lambda x, xp, fp: jax.numpy.interp(x=x, xp=xp, fp=fp)
    return jax.vmap(jax.vmap(in_fn, (None, None, 1), 0), (None, None, 2), 1)(x, xp, fp)
def get_coeff_file(coeff_file = None, verbose=False):
    """ if coeff_file is None, the script will search for the file in the
            same folder where this file is sorted to use the last one.
    """
    import glob,os
    this_file_folder = os.path.split(os.path.abspath(__file__))[0]
    coeff_files = sorted(glob.glob(os.path.join(this_file_folder,
                    'igrf??coeffs.txt')))
    if type(coeff_file) is type(None):
        # read the information from the file
        coeff_file = coeff_files[-1]
        if verbose:
            print("Using coefficients file:",os.path.basename(coeff_file))

    return coeff_file, coeff_files

coeff_file, coeff_files = get_coeff_file(verbose=True)


def read_coeff_file(coeff_file=None, verbose=False) -> Tuple[jp.ndarray, jp.ndarray, jp.ndarray, jp.ndarray, jp.ndarray]:
    """
    Reads the IGRF coefficients from coeff_file.
    """
    try:
        with open(coeff_file,'r') as fp:
            txtlines = fp.read().split('\n')
    except:
        raise IOError("Problems reading coefficients file:\n%s"%coeff_file)

    for line in reversed(txtlines): # start from the bottom to get largest n
        if len(line) < 3: continue # If line is too small skip
        max_n = int(line.split()[1]) # getting the largest n (13 in igrf11)
        if verbose:
            print("max_n is",max_n)
        break
    for line in txtlines:
        if len(line) < 3: continue # If line is too small skip
        if line[0:2] in ['g ', 'h ']: # reading the coefficients
            n = int(line.split()[1])
            m = int(line.split()[2])
            if line[0] == 'g':
                gdat[:,m,n] = jp.array(line.split()[3:], dtype=float) # type: ignore
            elif line[0] == 'h':
                hdat[:,m,n] = jp.array(line.split()[3:], dtype=float) # type: ignore
        elif line[0:3] == 'g/h': #reading the epochs
            all_epochs = line.split()[3:]
            secular_variation = all_epochs[-1]
            epoch = jp.array(all_epochs[:-1]+["2025"],dtype=float) # read the epochs
            gdat = jp.zeros([epoch.size,max_n + 1, max_n + 1],float) # type: ignore
            hdat = jp.zeros([epoch.size,max_n + 1, max_n + 1],float) # type: ignore
    gdat[-1,:,:] *= 5
    gdat[-1,:,:] += gdat[-2,:,:]
    hdat[-1,:,:] *= 5 
    hdat[-1,:,:] += hdat[-2,:,:]

    return max_n, gdat, hdat, epoch, secular_variation # type: ignore

max_n, gdat, hdat, epoch, secular_variation = read_coeff_file(coeff_file=coeff_file, verbose=True)



def get_m_n_schmidt(max_n: int) -> Tuple[jp.ndarray, jp.ndarray, jp.ndarray]:
    """
        Builds up the "schmidt" coefficients !!! careful with this definition
        Schmidt quasi-normalized associated Legendre functions of degree n
        and order m. Thebault et al. 2015
    """
    x = jp.array(list(range(max_n+1)))

    n, m= jax.numpy.meshgrid(x, x)

    schmidt = jp.sqrt(2 * factorial(n - m) / factorial(n + m)) * (-1) ** m
    schmidt = schmidt.at[0,:].set(1.)
    return m, n, jax.numpy.triu(schmidt)

m, n, schmidt = get_m_n_schmidt(max_n=max_n)







@jax.jit
def igrf_B(year: jp.ndarray, ht: jp.ndarray, lon: jp.ndarray, lat: jp.ndarray) -> Tuple[jp.ndarray, jp.ndarray, jp.ndarray, jp.ndarray]:
    """
    [Bn,Be,Bd,B] = igrf_B(year, ht, lon, lat)
    returns Bn Be Bd components of geomagnetic field based on igrf-X model
    and B=sqrt(Bn**2 + Be**2 + Bd**2), Bn=north, Be=east, Bd=down (nT),
    1900.<year<max_year.,
    ht: (km above Earth radius a),
    lon: (deg, east>0),
    lat: (deg, geocentric, north>0)
            note: geodetic coordinates should be translated to geocentric
                before calling this function.
    """

    g = interpolate(x=year, xp=epoch, fp=gdat)
    h = interpolate(x=year, xp=epoch, fp=hdat)
    a = a_igrf

    phi = lon*jp.pi/180.    # set phi=longitude dependence - co-sinusoids
    cp  = jp.cos(m * phi)
    sp  = jp.sin(m * phi)
    az  = g * cp + h * sp

    az_phi = m * (-g * sp + h * cp)

    r = a + ht        # set geocentric altitude dependence
    amp   = a * jax.numpy.power(a / r, n + 1)
    amp_r = -(n + 1) * amp / r                # r derivative of amp


    theta = (90. - lat) * jp.pi / 180.    # set theta=colatitude dependence
    ct = jp.cos(theta)
    st = jp.sqrt(1. - ct ** 2.)
    lPmn, lPmn_der = lpmn(max_n, max_n, ct)    # assoc legendre and derivative
    lPmn_der = jax.numpy.squeeze(lPmn_der)
    lPmn = jax.numpy.squeeze(lPmn)
    # print(jp.norm(lPmn)) # .shape)
    # print(lPmn_der.shape)
    lPmn = lPmn * schmidt    # schmidt normalization
    lPmn_theta = -st * lPmn_der * schmidt
    # print(lPmn_der) # .shape)
    # print(jp.norm(st))

    # print(jp.norm(schmidt))
    Z = jp.sum((amp_r * lPmn * az))       # get field components (nT)
    Y = -jp.sum((amp * lPmn * az_phi)) / (r * st)
    X = jp.sum((amp * lPmn_theta * az)) / r
    B = jp.sqrt(X ** 2. + Y ** 2. + Z ** 2.)

    return X,Y,Z,B



# %%

import numpy as np
import datetime
# Geodetic to Geocentric coordinates transformation
# WGS84 constants
# reference:
# http://earth-info.nga.mil/GandG/publications/tr8350.2/wgs84fin.pdf
a_WGS=6378.137
flatness = 1./298.257223563  # flatenning
b_WGS=a_WGS*(1.-flatness)    # WGS polar radius (semi-minor axis) in km
eccentricity=np.sqrt(a_WGS**2-b_WGS**2)/a_WGS

def geod2geoc(lon,geodlat,h):
    # returns geocentric xyz coordinates (ECEF) in km of a target with
    # latitude       geodlat (rad) --- geodetic
    # longitude      lon (rad)
    # height         h (km above local ellipsoid)
    n=a_WGS / np.sqrt(1.-flatness*(2.-flatness) * np.sin(geodlat)**2.)
    # cartesian geocentric coordinates wrt Greenwich
    x=(n+h)*np.cos(geodlat)*np.cos(lon)
    y=(n+h)*np.cos(geodlat)*np.sin(lon)
    z=(n*(1.-eccentricity**2.)+h)*np.sin(geodlat)   
    
    p   = np.sqrt(x**2. + y**2.)
    geoclat = np.arctan2(z,p)        # geocentric latitude (theta)
    
    return x,y,z,geoclat

def datetime2years(dt0):
    daysinyear = (dt0 - datetime.datetime(dt0.year,1,1)).total_seconds()/24/3600
    totaldaysinyear = datetime.datetime(dt0.year,12,31).timetuple().tm_yday
    return dt0.year + daysinyear/totaldaysinyear

geod_lat = 65.12992
lon = -147.47104
geod_ht = 100   #km above WGS-84 ellipsoide
print("geodetic coordinates: latitude:%g deg, longitude:%g deg, altitude:%g km"%(
    geod_lat,lon,geod_ht))
deg = 180/np.pi
x,y,z,geoc_lat = geod2geoc(lon/deg, geod_lat/deg, geod_ht) # convert to geocentric
ht_igrf = np.sqrt(x**2 + y**2 + z**2) - a_igrf # height above IGRF sphere with radius 6371.2 km
dt0 = datetime.datetime(2021,2,18,12,12,12)
year = datetime2years(dt0)

print("geocentric coordinates: latitude:%g deg, longitude:%g deg, IGRF altitude:%g km"%(
    geoc_lat*deg,lon,ht_igrf))

# %%
%timeit Bn,Be,Bd,B = igrf_B(jp.array(year), jp.array([ht_igrf]), jp.array([lon]), jp.array([geoc_lat*deg]))
# print("B = %g nT: %g nT northwards, %g nT eastwards, %g nT downwards"%(
#         B,Bn,Be,Bd))
# %%
