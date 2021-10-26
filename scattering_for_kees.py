import numpy as np
import shutil
import sys
from io import StringIO
import astropy.constants as c

import os
import tempfile
from pathlib import Path
import radmc3dPy
import disklab
from disklab.radmc3d import write

# define some constants

c_light = c.c.cgs.value
au = c.au.cgs.value
R_sun = c.R_sun.cgs.value
M_sun = c.M_sun.cgs.value
k_b = c.k_B.cgs.value
G = c.G.cgs.value
m_p = c.m_p.cgs.value
k_B = c.k_B.cgs.value
h = c.h.cgs.value


def bplanck(freq, temp):
    """
    This function computes the Planck function

                   2 h nu^3 / c^2
       B_nu(T)  = ------------------    [ erg / cm^2 s ster Hz ]
                  exp(h nu / kT) - 1

    Arguments:
     freq  [Hz]            = Frequency in Herz (can be array)
     temp  [K]             = Temperature in Kelvin (can be array)
    """
    const1 = h / k_B
    const2 = 2 * h / c_light**2
    const3 = 2 * k_B / c_light**2
    x = const1 * freq / (temp + 1e-99)
    if np.isscalar(x):
        if x > 500.:
            x = 500.
    else:
        x[np.where(x > 500.)] = 500.
    bpl = const2 * (freq**3) / ((np.exp(x) - 1.e0) + 1e-99)
    bplrj = const3 * (freq**2) * temp
    if np.isscalar(x):
        if x < 1.e-3:
            bpl = bplrj
    else:
        ii = x < 1.e-3
        bpl[ii] = bplrj[ii]
    return bpl


class Capturing(list):
    """Context manager capturing standard output of whatever is called in it.

    Keywords
    --------
    stderr : bool
        if True will capture the standard error instead of standard output.
        defaulats to False

    Examples
    --------
    >>> with Capturing() as output:
    >>>     do_something(my_object)

    `output` is now a list containing the lines printed by the function call.

    This can also be concatenated

    >>> with Capturing() as output:
    >>>    print('hello world')
    >>> print('displays on screen')
    displays on screen

    >>> with output:
    >>>     print('hello world2')
    >>> print('output:', output)
    output: ['hello world', 'hello world2']

    >>> import warnings
    >>> with output, Capturing(stderr=True) as err:
    >>>     print('hello world2')
    >>>     warnings.warn('testwarning')
    >>> print(output)
    output: ['hello world', 'hello world2']

    >>> print('error:', err[0].split(':')[-1])
    error:  testwarning

    Mostly copied from [this stackoverflow answer](http://stackoverflow.com/a/16571630/2108771)

    """

    def __init__(self, *args, stderr=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._error = stderr

    def __enter__(self):
        """Start capturing output when entering the context"""
        if self._error:
            self._std = sys.stderr
            sys.stderr = self._stringio = StringIO()
        else:
            self._std = sys.stdout
            sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        """Get & return the collected output when exiting context"""
        self.extend(self._stringio.getvalue().splitlines())
        if self._error:
            sys.stderr = self._std
        else:
            sys.stdout = self._std


def J_over_B(tauz_in, eps_e, tau):
    # our tauz goes from 0 to tau
    # while in the paper it goes from -tau/2 to +tau/2
    if isinstance(tauz_in, np.ndarray):
        tauz = tauz_in.copy() - tau / 2
    else:
        tauz = tauz_in - tau / 2

    b = 1.0 / (
        (1.0 - np.sqrt(eps_e)) * np.exp(-np.sqrt(3 * eps_e) * tau) + 1 + np.sqrt(eps_e))

    J = 1.0 - b * (
        np.exp(-np.sqrt(3.0 * eps_e) * (0.5 * tau - tauz)) +
        np.exp(-np.sqrt(3.0 * eps_e) * (0.5 * tau + tauz)))

    return J


def S_over_B(tauz, eps_e, tau):
    return eps_e + (1.0 - eps_e) * J_over_B(tauz, eps_e, tau)


def I_over_B_out(tau_total, eps_e, mu=1, ntau=300):
    """Integrates the scattering solution numerically

    Parameters
    ----------
    tau_total : float
        total optical depth
    eps_e : float
        effective extinction probablility (1-albedo)
    mu : float, optional
        cosine of the incidence angle, by default 1
    ntau : int, optional
        number of grid points, by default 300

    Returns
    -------
    float
        outgoing intensity in units of the planck function.
    """
    tau = np.linspace(0, tau_total, ntau)
    Inu = np.zeros(ntau)
    Jnu = J_over_B(tau, eps_e, tau_total)
    # the first 1.0 here is a placeholder for Bnu, just for reference
    Snu = eps_e * 1.0 + (1.0 - eps_e) * Jnu
    for i in range(1, ntau):
        dtau = (tau[i] - tau[i - 1]) / mu
        expdtray = np.exp(-dtau)
        srcav = 0.5 * (Snu[i] + Snu[i - 1])
        Inu[i] = expdtray * Inu[i - 1] + (1 - expdtray) * srcav
    return Inu[-1]


def I_over_B(eps_e, tau, mu=1):
    arg = np.where(np.array(tau) > 2. / 3. * mu, tau - 2. / 3. * mu, tau)
    # arg = tau - 2. / 3. * mu if tau > 2. / 3. * mu else tau
    return (1.0 - np.exp(-tau / mu)) * S_over_B(arg, eps_e, tau)
    # tauray = (tau[-1] - tau) / mu
    # I = (1.0 - np.exp(-tau / mu)) * np.interp(2. / 3., tauray[::-1], Snu[::-1])


def radmc3d_scattering_solution(tau, eps_e, T_fg, T_bg, n_phot=100000, keep_folder=False):
    """
    tau : float
        total optical depth (abs + sca) [-]

    eps_e : float
        effective absorption opacity [-]

    T_fg : float
        temperature of the medium [K]

    T_bg : float
        background temperature [K]
    """
    lam = 0.1  # arbitrary wavelength

    # run everything in a temporary directory that gets deleted in the end
    with tempfile.TemporaryDirectory(dir='.') as tempdir:
        setupdir = Path(tempdir)

        lam_mic = lam * 1e4

        h = 1.0
        n_z = 300
        zi = np.linspace(0, h, n_z + 1)  # interfaces
        z = 0.5 * (zi[1:] + zi[:-1])   # cell centers

        kap_sce = (1 - eps_e)
        kap_abs = eps_e

        kap_ext = kap_abs + kap_sce

        # rho     = 2 * Sigma / np.sqrt(2 * np.pi * h * h) * np.exp(-z**2/(2 * h * h)) # dust density [g/cm^3]
        rho = tau / (kap_ext * h) * np.ones(n_z)
        T_dust = T_fg * np.ones(n_z)     # temperature profile [K]

        one_two = np.array([1 - 1e-6, 1 + 1e-6])  # makes a 2 element wavelength range from one wavelength to avoid RADMC complaining about interpolation issues

        with open(setupdir / 'amr_grid.inp', 'w') as f:
            write(f, 1)                              # format identifier
            write(f, 0)                              # 0 = regular grid
            write(f, 10)                             # 10 = 1d plane parallel
            write(f, 0)                              # (always 0)
            write(f, 0, 0, 1)                        # include only z-coordinate
            write(f, 1, 1, n_z)                      # grid size
            write(f, -1e90)                          # x_min
            write(f, +1e90)                          # x_max
            write(f, -1e90)                          # y_min
            write(f, +1e90)                          # y_max
            write(f, *zi, sep='\n')  # the cell interface radii, adding large number to make it almost plane parallel

        with open(setupdir / 'dust_density.inp', 'w') as f:
            write(f, 1)                       # Format identifier
            write(f, n_z)                     # Number of cells
            write(f, 1)                       # Nr of dust species
            write(f, *rho, sep='\n')           # the density

        with open(setupdir / 'dust_temperature.dat', 'w') as f:
            write(f, 1)                       # Format identifier
            write(f, n_z)                     # Number of cells
            write(f, 1)                       # Nr of dust species
            write(f, *T_dust, sep='\n')       # the temperature

        with open(setupdir / 'wavelength_micron.inp', 'w') as f:
            write(f, 2)  # length
            write(f, *(lam_mic * one_two), sep='\n')

        with open(setupdir / 'mcmono_wavelength_micron.inp', 'w') as f:
            write(f, 1)  # length
            write(f, lam_mic, sep='\n')

        with open(setupdir / 'dustkappa_silicate.inp', 'w') as f:
            write(f, 3)  # format
            write(f, 2)  # length
            write(f, lam_mic * one_two[0], kap_abs, kap_sce, 0)
            write(f, lam_mic * one_two[1], kap_abs, kap_sce, 0)

        with open(setupdir / 'dustopac.inp', 'w') as f:
            write(f, '2               Format number of this file')
            write(f, '1               Nr of dust species')
            write(f, '============================================================================')
            write(f, '1               Way in which this dust species is read')
            write(f, '0               0=Thermal grain')
            write(f, 'silicate        Extension of name of dustkappa_***.inp file')
            write(f, '----------------------------------------------------------------------------')

        with open(setupdir / 'radmc3d.inp', 'w') as f:
            write(f, 'nphot = ', n_phot)
            write(f, 'nphot_mono = ', n_phot)
            write(f, 'nphot_scat = ', n_phot)
            write(f, 'scattering_mode = 1')
            write(f, f'thermal_boundary_zl = {T_bg}')

        # Calculate Intensity

        capt = Capturing()

        with capt:
            disklab.radmc3d.radmc3d('mcmono', path=setupdir)

        with open(setupdir / 'mean_intensity.out', 'r') as f:
            _iformat = np.fromfile(f, int, count=1, sep=' ')[0]  # noqa
            _nz = np.fromfile(f, int, count=1, sep=' ')[0]   # noqa
            _nnu = np.fromfile(f, int, count=1, sep=' ')[0]   # noqa
            _nu = np.fromfile(f, float, count=1, sep=' ')[0]   # noqa
            jnu = np.fromfile(f, float, sep=' ')   # noqa

        # Image

        with capt:
            disklab.radmc3d.radmc3d('image allwl incl 0', path=setupdir)

        # Read image

        im_value = np.nan
        if os.path.isfile(setupdir / 'image.out'):
            im_value = radmc3dPy.image.readImage(fname=setupdir / 'image.out').image.max()

        if keep_folder:
            shutil.copytree(setupdir, 'radmc3d_setup', dirs_exist_ok=True)

        return {
            'I_out': im_value,
            'z': z,
            'zi': zi,
            'rho': rho,
            'Jnu': jnu,
            'output': capt
        }
