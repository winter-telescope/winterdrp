import logging
import os
import astropy.io.fits
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from winterdrp.pipelines.base_pipeline import Pipeline
from winterdrp.downloader.caltech import download_via_ssh
from astropy.time import Time
from winterdrp.catalog import Gaia2Mass, PS1
from winterdrp.processors.database.database_exporter import DatabaseImageExporter
from winterdrp.processors.astromatic.sextractor.sextractor import sextractor_header_key
from winterdrp.processors.autoastrometry import AutoAstrometry
from winterdrp.processors.astromatic import Sextractor, Scamp, Swarp
from astropy.io import fits
from winterdrp.pipelines.summer.summer_files import get_summer_schema_path, summer_mask_path, summer_weight_path, \
    sextractor_astrometry_config, sextractor_photometry_config, scamp_path, swarp_path
from winterdrp.pipelines.summer.summer_files.schema import summer_schema_dir
from winterdrp.processors.split import SplitImage
from winterdrp.processors.utils import ImageSaver
from winterdrp.processors.utils.image_loader import ImageLoader
from winterdrp.processors.utils.image_selector import ImageSelector, ImageBatcher
from winterdrp.processors.photcal import PhotCalibrator
from winterdrp.processors import MaskPixels, BiasCalibrator, FlatCalibrator
from winterdrp.processors.csvlog import CSVLog
from winterdrp.paths import core_fields, base_name_key, latest_save_key

summer_flats_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
summer_gain = 1.0
summer_pixel_scale = 0.466

logger = logging.getLogger(__name__)


def summer_astrometric_catalog_generator(
        header: astropy.io.fits.Header
):
    temp_cat_path = header[sextractor_header_key]
    cat = Gaia2Mass(
        min_mag=10,
        max_mag=20,
        search_radius_arcmin=30,
        trim=True,
        image_catalog_path=temp_cat_path,
        filter_name='j'
    )
    return cat


def summer_photometric_catalog_generator(
        header: astropy.io.fits.Header
):
    filter_name = header['FILTERID']
    return PS1(min_mag=10, max_mag=20, search_radius_arcmin=30, filter_name=filter_name)


def load_raw_summer_image(
        path: str
) -> tuple[np.array, astropy.io.fits.Header]:
    with fits.open(path) as data:
        header = data[0].header
        header["OBSCLASS"] = ["calibration", "science"][header["OBSTYPE"] == "SCIENCE"]
        # print(header['OBSCLASS'])
        header['UTCTIME'] = header['UTCSHUT']
        header['TARGET'] = header['OBSTYPE'].lower()
        # header['TARGET'] = header['FIELDID']
        crd = SkyCoord(ra=data[0].header['RA'], dec=data[0].header['DEC'], unit=(u.deg, u.deg))
        header['RA'] = crd.ra.deg
        header['DEC'] = crd.dec.deg
        header['CRVAL1'] = header['RA']
        header['CRVAL2'] = header['DEC']
        tel_crd = SkyCoord(ra=data[0].header['TELRA'], dec=data[0].header['TELDEC'], unit=(u.deg, u.deg))
        header['TELRA'] = tel_crd.ra.deg
        header['TELDEC'] = tel_crd.dec.deg
        # filters = {'4': 'OPEN', '3': 'r', '1': 'u'}
        header['BZERO'] = 0
        header[latest_save_key] = path
        header["RAWPATH"] = path
        # print(img[0].data.shape)
        data[0].data = data[0].data * 1.0
        # img[0].data[2048, :] = np.nan

        if 'other' in header['FILTERID']:
            header['FILTERID'] = 'r'

        header["CALSTEPS"] = ""
        header["BASENAME"] = os.path.basename(path)
        header.append(('GAIN', summer_gain, 'Gain in electrons / ADU'), end=True)

        header['OBSDATE'] = int(header['UTC'].split('_')[0])

        obstime = Time(header['UTCISO'], format='iso')
        t0 = Time('2018-01-01', format='iso')
        header['OBSID'] = 0
        header['NIGHT'] = np.floor((obstime - t0).jd).astype(int)
        header['PROGID'] = 0
        header['EXPMJD'] = header['OBSMJD']

        if "SUBPROG" not in header.keys():
            header['SUBPROG'] = 'high_cadence'

        header['FILTER'] = header['FILTERID']
        header['DARKNAME'] = ''
        # print('Time', header['shutopen'])
        try:
            header['SHUTOPEN'] = Time(header['SHUTOPEN'], format='iso').jd
        except (KeyError, ValueError):
            # header['SHUTOPEN'] = None
            pass

        try:
            header['SHUTCLSD'] = Time(header['SHUTCLSD'], format='iso').jd
        except ValueError:
            pass
            # header['SHUTCLSD'] = None

        header['PROCFLAG'] = 0
        sunmoon_keywords = ['MOONRA', 'MOONDEC', 'MOONILLF', 'MOONPHAS', 'MOONALT', 'SUNAZ', 'SUNALT']
        for key in sunmoon_keywords:
            val = 0
            if key in header.keys():
                if header[key] not in ['']:
                    val = header[key]
            header[key] = val

        itid_dict = {
            'SCIENCE': 1,
            'BIAS': 2,
            'FLAT': 2,
            'DARK': 2,
            'FOCUS': 3,
            'POINTING': 4,
            'OTHER': 5
        }

        if not header['OBSTYPE'] in itid_dict.keys():
            header['ITID'] = 5
        else:
            header['ITID'] = itid_dict[header['OBSTYPE']]

        if header['FIELDID'] == 'radec':
            header['FIELDID'] = 0

        if header['ITID'] != 1:
            header['FIELDID'] = -99

        if 'COADDS' not in header.keys():
            header['COADDS'] = 1
            # logger.debug('Setting COADDS to 1')

        crds = SkyCoord(ra=header['RA'], dec=header['DEC'], unit=(u.deg, u.deg))
        header['RA'] = crds.ra.deg
        header['DEC'] = crds.dec.deg

        data[0].header = header
    return data[0].data, data[0].header


pipeline_name = "summer"


class SummerPipeline(Pipeline):
    name = pipeline_name

    pipeline_configurations = {
        None: [
            ImageLoader(
                load_image=load_raw_summer_image
            ),
            CSVLog(
                export_keys=[
                                "UTC", 'FIELDID', "FILTERID", "EXPTIME", "OBSTYPE", "RA", "DEC", "TARGTYPE",
                                base_name_key
                            ] + core_fields
            ),
            DatabaseImageExporter(
                db_name=pipeline_name,
                db_table="exposures",
                schema_path=get_summer_schema_path("exposures"),
                full_setup=True,
                schema_dir=summer_schema_dir
            ),
            MaskPixels(mask_path=summer_mask_path),
            # SplitImage(
            #     buffer_pixels=0,
            #     n_x=1,
            #     n_y=2
            # ),
            # ImageSaver(output_dir_name="rawimages"),
            DatabaseImageExporter(
                db_name=pipeline_name,
                db_table="raw",
                schema_path=get_summer_schema_path("raw")
            ),
            BiasCalibrator(),
            ImageBatcher(split_key="filter"),
            FlatCalibrator(),
            ImageSelector(
                (base_name_key, "SUMMER_20220402_214324_Camera0.fits"),
            ),
            ImageSaver(output_dir_name="scienceimages"),
            # ImageBatcher(split_key=["RA", "DEC"]),
            AutoAstrometry(pa=0, inv=True, pixel_scale=summer_pixel_scale),
            ImageSaver(output_dir_name="testb"),
            Sextractor(
                output_sub_dir="testb",
                weight_image=summer_weight_path,
                checkimage_name=None,
                checkimage_type=None,
                **sextractor_astrometry_config
            ),
            ImageSaver(output_dir_name="testc"),
            Scamp(
                ref_catalog_generator=summer_astrometric_catalog_generator,
                scamp_config_path=scamp_path,
            ),
            ImageSaver(output_dir_name="testd"),
            Swarp(swarp_config_path=swarp_path, imgpixsize=2400),
            ImageSaver(output_dir_name="photprocess"),
            Sextractor(output_sub_dir="photprocess",
                       checkimage_name='NONE',
                       checkimage_type='NONE',
                       **sextractor_photometry_config),
            ImageSaver(output_dir_name="processed"),
            PhotCalibrator(ref_catalog_generator=summer_photometric_catalog_generator),
            ImageSaver(output_dir_name="processed", additional_headers=['PROCIMG']),
            DatabaseImageExporter(
                db_name=pipeline_name,
                db_table="proc",
                schema_path=get_summer_schema_path("proc")
            )

        ]
    }

    @staticmethod
    def download_raw_images_for_night(
            night: str | int
    ):
        download_via_ssh(
            server="jagati.caltech.edu",
            base_dir="/data/viraj/winter_data/commissioning/raw/",
            night=night,
            pipeline=pipeline_name
        )

    # @staticmethod
    # def load_raw_image(
    #         path: str
    # ) -> tuple[np.array, astropy.io.fits.Header]:
    #     with fits.open(path) as data:
    #         header = data[0].header
    #         header["OBSCLASS"] = ["calibration", "science"][header["OBSTYPE"] == "SCIENCE"]
    #         # print(header['OBSCLASS'])
    #         header['UTCTIME'] = header['UTCSHUT']
    #         header['TARGET'] = header['OBSTYPE'].lower()
    #         # header['TARGET'] = header['FIELDID']
    #         crd = SkyCoord(ra=data[0].header['RA'], dec=data[0].header['DEC'], unit=(u.deg, u.deg))
    #         header['RA'] = crd.ra.deg
    #         header['DEC'] = crd.dec.deg
    #         header['CRVAL1'] = header['RA']
    #         header['CRVAL2'] = header['DEC']
    #         tel_crd = SkyCoord(ra=data[0].header['TELRA'], dec=data[0].header['TELDEC'], unit=(u.deg, u.deg))
    #         header['TELRA'] = tel_crd.ra.deg
    #         header['TELDEC'] = tel_crd.dec.deg
    #         # filters = {'4': 'OPEN', '3': 'r', '1': 'u'}
    #         header['BZERO'] = 0
    #
    #         # print(img[0].data.shape)
    #         data[0].data = data[0].data * 1.0
    #         # img[0].data[2048, :] = np.nan
    #
    #         if 'other' in header['FILTERID']:
    #             header['FILTERID'] = 'r'
    #
    #         header["CALSTEPS"] = ""
    #         header["BASENAME"] = os.path.basename(path)
    #         header.append(('GAIN', summer_gain, 'Gain in electrons / ADU'), end=True)
    #
    #         header['OBSDATE'] = int(header['UTC'].split('_')[0])
    #
    #         obstime = Time(header['UTCISO'], format='iso')
    #         t0 = Time('2018-01-01', format='iso')
    #         header['OBSID'] = 0
    #         header['NIGHT'] = np.floor((obstime - t0).jd).astype(int)
    #         header['PROGID'] = 0
    #         header['EXPMJD'] = header['OBSMJD']
    #
    #         if "SUBPROG" not in header.keys():
    #             header['SUBPROG'] = 'high_cadence'
    #
    #         header['FILTER'] = header['FILTERID']
    #
    #         # print('Time', header['shutopen'])
    #         try:
    #             header['SHUTOPEN'] = Time(header['SHUTOPEN'], format='iso').jd
    #         except (KeyError, ValueError):
    #             # header['SHUTOPEN'] = None
    #             pass
    #
    #         try:
    #             header['SHUTCLSD'] = Time(header['SHUTCLSD'], format='iso').jd
    #         except ValueError:
    #             pass
    #             # header['SHUTCLSD'] = None
    #
    #         header['PROCFLAG'] = 0
    #         sunmoon_keywords = ['MOONRA', 'MOONDEC', 'MOONILLF', 'MOONPHAS', 'MOONALT', 'SUNAZ', 'SUNALT']
    #         for key in sunmoon_keywords:
    #             val = 0
    #             if key in header.keys():
    #                 if header[key] not in ['']:
    #                     val = header[key]
    #             header[key] = val
    #
    #         itid_dict = {
    #             'SCIENCE': 1,
    #             'BIAS': 2,
    #             'FLAT': 2,
    #             'DARK': 2,
    #             'FOCUS': 3,
    #             'POINTING': 4,
    #             'OTHER': 5
    #         }
    #
    #         if not header['OBSTYPE'] in itid_dict.keys():
    #             header['ITID'] = 5
    #         else:
    #             header['ITID'] = itid_dict[header['OBSTYPE']]
    #
    #         if header['FIELDID'] == 'radec':
    #             header['FIELDID'] = 0
    #
    #         if header['ITID'] != 1:
    #             header['FIELDID'] = -99
    #
    #         crds = SkyCoord(ra=header['RA'], dec=header['DEC'], unit=(u.deg, u.deg))
    #         header['RA'] = crds.ra.deg
    #         header['DEC'] = crds.dec.deg
    #
    #         data[0].header = header
    #     return data[0].data, data[0].header
