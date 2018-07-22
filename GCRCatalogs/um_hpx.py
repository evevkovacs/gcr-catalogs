"""
UM galaxy catalog class.
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

__all__ = ['UMGalaxyCatalog']
__version__ = '0.0.0'


class UMGalaxyCatalog(BaseGenericCatalog):
    """
    UM galaxy catalog class. Uses generic quantity and filter mechanisms
    defined by BaseGenericCatalog class.
    """

    def _subclass_init(self, catalog_root_dir, catalog_path_template, cosmology, healpix_pixels=None, zlo=None, zhi=None, check_file_metadata=False, **kwargs): 

        assert(os.path.isdir(catalog_root_dir)), 'Catalog directory {} does not exist'.format(catalog_root_dir)
        self._catalog_path_template = os.path.join(catalog_root_dir, catalog_path_template)

        self._native_filter_quantities = {'healpix_pixel', 'redshift_block_lower'}

        self._default_zrange_lo, self._default_zrange_hi, self._default_healpix_pixels = self._get_healpix_info()
        self.zrange_lo = self._default_zrange_lo if zlo is None else zlo
        self.zrange_hi = self._default_zrange_hi if zhi is None else zhi
        self.healpix_pixels = self._default_healpix_pixels if healpix_pixels is None else healpix_pixels
        self.check_healpix_file_list()                                                       
        self.cosmology = FlatLambdaCDM(**cosmology)
        self.version = kwargs.get('version', '0.0.0')
        self.check_file_metadata = check_file_metadata

        #get sky area and check files if requested
        sky_area = 0.
        for healpix in self.healpix_pixels:
            for zlo in range(self.zrange_lo, self.zrange_hi):
                healpix_file = self._catalog_path_template.format(zlo, zlo+1, healpix)
                fh = h5py.File(healpix_file, 'r')
                if check_file_metadata:
                    self._check_file_metadata(fh)
                if 'skyArea' in fh['metaData'].keys():
                    sky_area = sky_area + float(fh['metaData/skyArea'].value)
                fh.close()

        self.sky_area = sky_area if sky_area > 0 else np.nan

        # specify quantity modifiers
        self._quantity_modifiers = {
            'galaxy_id' :    'galaxy_id',
            'ra_true':       'ra',
            'dec_true':      'dec',
            'redshift_true': 'redshift',
            'halo_id':       'target_halo_id',
            'halo_mass':     'target_halo_mass',
            'stellar_mass':  'obs_sm',
            'position_x': 'x',
            'position_y': 'y',
            'position_z': 'z',
            'velocity_x': 'vx',
            'velocity_y': 'vy',
            'velocity_z': 'vz',
            'is_central': (lambda x: x == -1, 'upid'),
        }

        # add magnitudes
        for band in 'gri':
            self._quantity_modifiers['Mag_true_{}_sdss_z0'.format(band)] = 'restframe_extincted_sdss_abs_mag{}'.format(band)
            self._quantity_modifiers['Mag_true_{}_lsst_z0'.format(band)] = 'restframe_extincted_sdss_abs_mag{}'.format(band)

    def _get_healpix_info(self):
        path = self._catalog_path_template
        fname_pattern = os.path.basename(path).format('\d', '\d', '\d+') #include z and healpix #s in pattern
        pattern = re.compile('\d+')
        path = os.path.dirname(path)
     
        healpix_pixels = set()
        zvalues = set()
        for f in sorted(os.listdir(path)):                                                                                
            m = re.match(fname_pattern, f)                                                                        
            if m is not None:
                healpix_name = os.path.splitext(m.group())[0]
                zlo, zhi, hpx = pattern.findall(healpix_name)
                healpix_pixels.add(int(hpx))
                zvalues.add(int(zlo))
                zvalues.add(int(zhi))

        return min(zvalues), max(zvalues), list(healpix_pixels)


    def check_healpix_file_list(self):
        self.healpix_pixel_files = [self._catalog_path_template.format(z,z+1,h) for z in range(self.zrange_lo, self.zrange_hi) for h in self.healpix_pixels]
        assert all(os.path.isfile(f) for f in self.healpix_pixel_files), 'Problem with some catalog files'


    def _check_file_metadata(self, fh, tol=1e-4):

        catalog_version = list()
        for version_label in ('Major', 'Minor', 'MinorMinor'):
            try:
                catalog_version.append(fh['/metaData/version' + version_label].value)
            except KeyError:
                break
        catalog_version = StrictVersion('.'.join(map(str, catalog_version or (0, 0))))

        #check cosmology
        metakeys = fh['metaData'].keys()
        if 'H_0' in metakeys and 'Omega_matter' in metakeys and 'Omega_b' in metakeys:
            H0 = fh['metaData/H_0'].value
            Om0 = fh['metaData/Omega_matter'].value
            Ob0 = fh['metaData/Omega_b'].value
            if  abs(H0 - self.cosmology.H0.value) > tol or abs(Om0 - self.cosmology.Om0) > tol or abs(Ob0 - self.cosmology.Ob0) > tol:
                raise ValueError('Mismatch in cosmological parameters (H0:{}, Om0:{}, Ob0:{}) for healpix file {}'.format(H0, Om0, Ob0, healpix_file))

        # check versions
        config_version = StrictVersion(self.version)
        if config_version != catalog_version:
            raise ValueError('Catalog file for file {} version {} does not match config version {}'.format(healpix_file, catalog_version, config_version))
        if StrictVersion(__version__) < config_version:
            raise ValueError('Reader version {} is less than config version {}'.format(__version__, catalog_version))


    def _generate_native_quantity_list(self):
        #use first file in list to get information
        filename = self._catalog_path_template.format(self.zrange_lo, self.zrange_lo+1, self.healpix_pixels[0])
        with h5py.File(filename, 'r') as fh:
            for k in fh:
                if k.isdigit():
                    return list(fh[k].keys())


    def _iter_native_dataset(self, native_filters=None):
        for healpix in self.healpix_pixels:

            if native_filters is not None:
                fargs = dict(healpix_pixel=healpix)
                if not all(f[0](*(fargs[k] for k in f[1:])) for f in native_filters if set(f[1:]) == set(fargs)):
                    continue

            for zlo in range(self.zrange_lo, self.zrange_hi):

                if native_filters is not None:
                    fargs = dict(healpix_pixel=healpix, redshift_block_lower=zlo)
                    if not all(f[0](*(fargs[k] for k in f[1:])) for f in native_filters):
                        continue

                healpix_file = self._catalog_path_template.format(zlo, zlo+1, healpix)
                with h5py.File(healpix_file, 'r') as fh:
                    for k in fh:
                        if k.isdigit():
                            yield lambda native_quantity: fh[k][native_quantity].value


    def _get_native_quantity_info_dict(self, quantity, default=None):
        #use first file in list to get information
        filename = self._catalog_path_template.format(self.zrange_lo, self.zrange_lo+1, self.healpix_pixels[0])
        with h5py.File(filename, 'r') as fh:
            quantity_key = '{}/{}'.format(fh.keys()[0], quantity) #use first lc shell
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