import astropy.io.fits
import numpy as np
import os
import logging
from winterdrp.processors.base_processor import BaseImageProcessor, PrerequisiteError
from winterdrp.paths import get_output_dir, copy_temp_file
from collections.abc import Callable
from winterdrp.catalog.base_catalog import BaseCatalog
from winterdrp.processors.astromatic.sextractor.sextractor import Sextractor, sextractor_header_key
from winterdrp.utils.ldac_tools import get_table_from_ldac
from astropy.coordinates import SkyCoord
import astropy.units as u
from winterdrp.errors import ProcessorError
from astropy.stats import sigma_clip, sigma_clipped_stats

logger = logging.getLogger(__name__)

# All the Sextractor parameters required for this script to run
REQUIRED_PARAMETERS = [
    "X_IMAGE",
    "Y_IMAGE",
    "FWHM_WORLD",
    'FLAGS',
    'ALPHAWIN_J2000',
    'DELTAWIN_J2000',
    'MAG_APER',
    'MAG_AUTO'
]


class PhotCalibrator(BaseImageProcessor):
    base_key = 'photcalibrator'

    def __init__(self,
                 ref_catalog_generator: Callable[[astropy.io.fits.Header], BaseCatalog],
                 temp_output_sub_dir: str = "phot",
                 redo: bool = True,
                 x_lower_limit: float = 100,
                 x_upper_limit: float = 2800,  # Are these floats or ints?
                 y_lower_limit: float = 100,
                 y_upper_limit: float = 2800,
                 fwhm_threshold_arcsec: float = 4.0,
                 num_matches_threshold: int = 5,
                 *args,
                 **kwargs):
        super(PhotCalibrator, self).__init__(*args, **kwargs)
        self.redo = redo  # What is this for?
        self.ref_catalog_generator = ref_catalog_generator
        self.temp_output_sub_dir = temp_output_sub_dir
        self.x_lower_limit = x_lower_limit
        self.x_upper_limit = x_upper_limit
        self.y_lower_limit = y_lower_limit
        self.y_upper_limit = y_upper_limit
        self.fwhm_threshold_arcsec = fwhm_threshold_arcsec # Why is this here not in catalog?
        self.num_matches_threshold = num_matches_threshold

    def get_phot_output_dir(self):
        return get_output_dir(self.temp_output_sub_dir, self.night_sub_dir)

    def calculate_zeropoint(
            self,
            ref_cat_path: str,
            img_cat_path: str
    ) -> list[dict]:
        ref_cat = get_table_from_ldac(ref_cat_path)
        img_cat = get_table_from_ldac(img_cat_path)

        if len(ref_cat) == 0:
            err = 'No sources found in reference catalog'
            logger.error(err)
            raise ProcessorError(err)

        ref_coords = SkyCoord(ra=ref_cat['ra'], dec=ref_cat['dec'], unit=(u.deg, u.deg))

        clean_mask = (img_cat['FLAGS'] == 0) & \
                     (img_cat['FWHM_WORLD'] < self.fwhm_threshold_arcsec / 3600.) & \
                     (img_cat['X_IMAGE'] > self.x_lower_limit) & \
                     (img_cat['X_IMAGE'] < self.x_upper_limit) & \
                     (img_cat['Y_IMAGE'] > self.y_lower_limit) & \
                     (img_cat['Y_IMAGE'] < self.y_upper_limit)

        clean_img_cat = img_cat[clean_mask]
        logger.debug(f'Found {len(clean_img_cat)} clean sources in image.')

        clean_img_coords = SkyCoord(ra=clean_img_cat['ALPHAWIN_J2000'], dec=clean_img_cat['DELTAWIN_J2000'],
                                    unit=(u.deg, u.deg))

        if 0 == len(clean_img_coords):
            err = 'No clean sources found in image'
            logger.error(err)
            raise ProcessorError(err)

        idx, d2d, d3d = ref_coords.match_to_catalog_sky(clean_img_coords)
        match_mask = d2d < 1.0 * u.arcsec
        matched_ref_cat = ref_cat[match_mask]
        matched_img_cat = clean_img_cat[idx[match_mask]]
        logger.info(f'Cross-matched {len(matched_img_cat)} sources from catalog to the image.')

        if len(matched_img_cat) < self.num_matches_threshold:
            err = f'Not enough cross-matched sources found to calculate a reliable zeropoint.'
            logger.error(err)
            raise ProcessorError(err)

        apertures = self.get_sextractor_apetures()  # aperture diameters
        zeropoints = []

        for i, aperture in enumerate(apertures):

            offsets = np.ma.array(matched_ref_cat['magnitude'] - matched_img_cat['MAG_APER'][:, i])
            cl_offset = sigma_clip(offsets)
            num_stars = np.sum(np.invert(cl_offset.mask))

            zp_mean, zp_med, zp_std = sigma_clipped_stats(offsets)

            check = [np.isnan(x) for x in [zp_mean, zp_med, zp_std]]
            if np.sum(check) > 0:
                err = f"Error with nan when calculating sigma stats: \n " \
                      f"mean: {zp_mean}, median: {zp_med}, std: {zp_std}"
                logger.error(err)
                raise ProcessorError(err)

            zero_dict = {'diameter': aperture, 'zp_mean': zp_mean, 'zp_median': zp_med, 'zp_std': zp_std,
                         'nstars': num_stars, 'mag_cat': matched_ref_cat['magnitude'][np.invert(cl_offset.mask)],
                         'mag_apers': matched_img_cat['MAG_APER'][:, i][np.invert(cl_offset.mask)]}
            zeropoints.append(zero_dict)

        offsets = np.ma.array(matched_ref_cat['magnitude'] - matched_img_cat['MAG_AUTO'])
        cl_offset = sigma_clip(offsets, sigma=3)
        num_stars = np.sum(np.invert(cl_offset.mask))
        zp_mean, zp_med, zp_std = sigma_clipped_stats(offsets, sigma=3)
        zero_auto_mag_cat = matched_ref_cat['magnitude'][np.invert(cl_offset.mask)]
        zero_auto_mag_img = matched_img_cat['MAG_AUTO'][np.invert(cl_offset.mask)]
        zeropoints.append({'diameter': 'AUTO', 'zp_mean': zp_mean, 'zp_median': zp_med, 'zp_std': zp_std,
                           'nstars': num_stars, 'mag_cat': zero_auto_mag_cat,
                           'mag_apers': zero_auto_mag_img
                           })

        return zeropoints

    def _apply_to_images(
            self,
            images: list[np.ndarray],
            headers: list[astropy.io.fits.Header],
    ) -> tuple[list[np.ndarray], list[astropy.io.fits.Header]]:

        phot_output_dir = self.get_phot_output_dir()

        try:
            os.makedirs(phot_output_dir)
        except OSError:
            pass

        for header in headers:
            ref_catalog = self.ref_catalog_generator(header)
            ref_cat_path = ref_catalog.write_catalog(header, output_dir=phot_output_dir)
            temp_cat_path = copy_temp_file(
                output_dir=phot_output_dir,
                file_path=header[sextractor_header_key]
            )

            zp_dicts = self.calculate_zeropoint(ref_cat_path, temp_cat_path)

            for zpvals in zp_dicts:
                header['ZP_%s' % (zpvals['diameter'])] = zpvals['zp_mean']
                header['ZP_%s_std' % (zpvals['diameter'])] = zpvals['zp_std']
                header['ZP_%s_nstars' % (zpvals['diameter'])] = zpvals['nstars']

            fwhm_med, fwhm_mean, fwhm_std = self.get_fwhm(temp_cat_path)
            header['FWHM_MED'] = fwhm_med
            header['FWHM_STD'] = fwhm_std
        return images, headers

    @staticmethod
    def get_fwhm(img_cat_path):
        imcat = get_table_from_ldac(img_cat_path)
        nemask = (imcat['X_IMAGE'] > 50) & (imcat['X_IMAGE'] < 2000) & (imcat['Y_IMAGE'] > 50) & (
                    imcat['Y_IMAGE'] < 2000)
        imcat = imcat[nemask]
        med_fwhm = np.median(imcat['FWHM_WORLD'])
        mean_fwhm = np.mean(imcat['FWHM_WORLD'])
        std_fwhm = np.std(imcat['FWHM_WORLD'])
        return med_fwhm, mean_fwhm, std_fwhm

    def get_sextractor_module(self) -> Sextractor:
        mask = [isinstance(x, Sextractor) for x in self.preceding_steps]
        return np.array(self.preceding_steps)[mask][-1]

    def check_prerequisites(
            self,
    ):

        mask = [isinstance(x, Sextractor) for x in self.preceding_steps]
        if np.sum(mask) < 1:
            err = f"{self.__module__} requires {Sextractor} as a prerequisite. " \
                  f"However, the following steps were found: {self.preceding_steps}."
            logger.error(err)
            raise PrerequisiteError(err)

        sextractor_param_path = self.get_sextractor_module().parameters_name

        logger.debug(f"Checking file {sextractor_param_path}")

        with open(sextractor_param_path, "rb") as f:
            sextractor_params = [x.strip().decode() for x in f.readlines() if len(x.strip()) > 0]
            sextractor_params = [x.split("(")[0] for x in sextractor_params if x[0] not in ["#"]]

        for param in REQUIRED_PARAMETERS:
            if param not in sextractor_params:
                err = f"Missing parameter: {self.__module__} requires {param} to run, " \
                      f"but this parameter was not found in sextractor config file '{sextractor_param_path}' . " \
                      f"Please add the parameter to this list!"
                logger.error(err)
                raise PrerequisiteError(err)

    def get_sextractor_apetures(self) -> list[float]:
        sextractor_config_path = self.get_sextractor_module().config

        with open(sextractor_config_path, "rb") as f:
            apeture_lines = [
                x.decode() for x in f.readlines() if np.logical_and(b"PHOT_APERTURES" in x, x.decode()[0] != "#")
            ]

        if len(apeture_lines) > 1:
            err = f"The config file {sextractor_config_path} has multiple entries for PHOT_APETURES."
            logger.error(err)
            raise ProcessorError(err)

        line = apeture_lines[0].replace("PHOT_APERTURES", " ").split("#")[0]

        return [float(x) for x in line.split(",") if x not in [""]]