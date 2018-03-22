"""
UM+SDSS galaxy catalog class.
"""
from __future__ import division
import os
import re
import warnings
import hashlib
from distutils.version import StrictVersion # pylint: disable=no-name-in-module,import-error
import numpy as np
import h5py
from astropy.cosmology import FlatLambdaCDM
from GCR import BaseGenericCatalog

__all__ = ['UMSDSSGalaxyCatalog']
__version__ = '3.0.0'


def md5(fname, chunk_size=65536):
    """
    generate MD5 sum for *fname*
    """
    hash_md5 = hashlib.md5()
    with open(fname, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _calc_conv(mag, shear1, shear2):
    slct = mag < 0.2
    mag_corr = np.copy(mag)
    mag_corr[slct] = 1.0 # manually changing the values for when magnification is near zero.
    conv = 1.0 - np.sqrt(1.0/mag_corr + shear1**2 + shear2**2)
    return conv


class UMSDSSGalaxyCatalog(BaseGenericCatalog):
    """
    UM+SDSS galaxy catalog class. Uses generic quantity and filter mechanisms
    defined by BaseGenericCatalog class.
    """

    def _subclass_init(self, filename, **kwargs): #pylint: disable=W0221

        assert os.path.isfile(filename), 'Catalog file {} does not exist'.format(filename)
        self._file = filename

        if kwargs.get('md5'):
            assert md5(self._file) == kwargs['md5'], 'md5 sum does not match!'
        else:
            warnings.warn('No md5 sum specified in the config file')

        self.lightcone = kwargs.get('lightcone')

        with h5py.File(self._file, 'r') as fh:
            # pylint: disable=no-member
            # get version
            catalog_version = list()
            for version_label in ('Major', 'Minor', 'MinorMinor'):
                try:
                    catalog_version.append(fh['/metaData/version' + version_label].value)
                except KeyError:
                    break
            catalog_version = StrictVersion('.'.join(map(str, catalog_version or (2, 0))))

            # get cosmology
            self.cosmology = FlatLambdaCDM(
                H0=fh['metaData/simulationParameters/H_0'].value,
                Om0=fh['metaData/simulationParameters/Omega_matter'].value,
                Ob0=fh['metaData/simulationParameters/Omega_b'].value,
            )

            # get sky area
            if catalog_version >= StrictVersion("2.1.1"):
                self.sky_area = float(fh['metaData/skyArea'].value)
            else:
                self.sky_area = 25.0 #If the sky area isn't specified use the default value of the sky area.

            # get native quantities
            self._native_quantities = set()
            def _collect_native_quantities(name, obj):
                if isinstance(obj, h5py.Dataset):
                    self._native_quantities.add(name)
            fh['galaxyProperties'].visititems(_collect_native_quantities)

        # check versions
        self.version = kwargs.get('version', '0.0.0')
        config_version = StrictVersion(self.version)
        if config_version != catalog_version:
            raise ValueError('Catalog file version {} does not match config version {}'.format(catalog_version, config_version))
        if StrictVersion(__version__) < config_version:
            raise ValueError('Reader version {} is less than config version {}'.format(__version__, catalog_version))

        # specify quantity modifiers
        self._quantity_modifiers = {
            'galaxy_id' :    'lightcone_id',
            'ra':            'ra_lensed',
            'dec':           'dec_lensed',
            'ra_true':       'ra',
            'dec_true':      'dec',
            'redshift_true': 'redshift',
            'shear_1':       'shear1',
            'shear_2':       'shear2',
            'convergence': (
                _calc_conv,
                'magnification',
                'shear1',
                'shear2',
            ),
            'magnification': (lambda mag: np.where(mag < 0.2, 1.0, mag), 'magnification'),
            'halo_id':       'target_halo_id',
            'halo_mass':     'target_halo_mass',
            'stellar_mass':  'obs_sm',
            'position_x': 'x',
            'position_y': 'y',
            'position_z': 'z',
            'velocity_x': 'vx',
            'velocity_y': 'vy',
            'velocity_z': 'vz',
        }

        # add magnitudes
        for band in 'gri':
            self._quantity_modifiers['Mag_true_{}_sdss_z0'.format(band)] = 'restframe_extincted_sdss_abs_mag{}'.format(band)


    def _generate_native_quantity_list(self):
        return self._native_quantities


    def _iter_native_dataset(self, native_filters=None):
        assert not native_filters, '*native_filters* is not supported'
        with h5py.File(self._file, 'r') as fh:
            def _native_quantity_getter(native_quantity):
                return fh['galaxyProperties/{}'.format(native_quantity)].value # pylint: disable=no-member
            yield _native_quantity_getter


    def _get_native_quantity_info_dict(self, quantity, default=None):
        with h5py.File(self._file, 'r') as fh:
            quantity_key = 'galaxyProperties/' + quantity
            if quantity_key not in fh:
                return default
            modifier = lambda k, v: None if k == 'description' and v == b'None given' else v.decode()
            return {k: modifier(k, v) for k, v in fh[quantity_key].attrs.items()}


    def _get_quantity_info_dict(self, quantity, default=None):
        q_mod = self.get_quantity_modifier(quantity)
        if callable(q_mod) or (isinstance(q_mod, (tuple, list)) and len(q_mod) > 1 and callable(q_mod[0])):
            warnings.warn('This value is composed of a function on native quantities. So we have no idea what the units are')
            return default
        return self._get_native_quantity_info_dict(q_mod or quantity, default=default)
