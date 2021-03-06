import os
import shutil

from astropy.io import fits
from glob import glob
import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

base_raw_dir = os.getenv("RAW_DATA_DIR")

if base_raw_dir is None:
    err = "No raw data directory specified. Run 'export RAW_DATA_DIR=/path/to/data' to set. " \
          "The raw data directory will need to be specified manually for path function."
    logger.warning(err)

base_output_dir = os.getenv("OUTPUT_DATA_DIR")

if base_output_dir is None:
    err = "No output data directory specified. Run 'export OUTPUT_DATA_DIR=/path/to/data' to set this. " \
          "The output directory will need to be specified manually for path functions."
    logger.warning(err)


raw_img_sub_dir = "raw"


def raw_img_dir(
        sub_dir: str = "",
        raw_dir: str = base_raw_dir
) -> str:
    return os.path.join(raw_dir, os.path.join(str(sub_dir), raw_img_sub_dir))


def get_preprocess_path(
        raw_img_path: str
) -> str:
    return raw_img_path.replace("/raw/", "/preprocess/")


def get_output_dir(
        dir_root: str,
        sub_dir: str | int = "",
        output_dir: str = base_output_dir
) -> str:
    return os.path.join(output_dir, os.path.join(str(sub_dir), dir_root))


def get_output_path(
        base_name: str,
        dir_root: str,
        sub_dir: str | int = "",
        output_dir: str = base_output_dir
) -> str:
    return os.path.join(get_output_dir(dir_root, sub_dir=str(sub_dir), output_dir=output_dir), base_name)


def cal_output_dir(
        sub_dir: str | int = "",
        output_dir: str = base_output_dir
) -> str:
    return get_output_dir("calibration", sub_dir=str(sub_dir), output_dir=output_dir)


cal_output_sub_dir = "calibration"


def reduced_img_dir(
        sub_dir: str | int = "",
        output_dir: str = base_output_dir
) -> str:
    return get_output_dir("redux", sub_dir=str(sub_dir), output_dir=output_dir)


def reduced_img_path(
        img_name: str,
        sub_dir: str | int = "",
        output_dir: str = base_output_dir
) -> str:
    return os.path.join(reduced_img_dir(str(sub_dir), output_dir=output_dir), img_name)


def astrometry_output_dir(
        sub_dir: str = "",
        astro_pass: int = 1,
        output_dir: str = base_output_dir
) -> str:
    return get_output_dir(f"astrometry_{astro_pass}", sub_dir=str(sub_dir), output_dir=output_dir)


def get_mask_path(
        img_path: str,
) -> str:
    return os.path.splitext(img_path)[0] + ".mask.fits"


def get_temp_path(
        output_dir: str,
        file_path: str
) -> str:
    return os.path.join(output_dir, "temp_" + os.path.basename(file_path))


def get_untemp_path(
        temp_path: str
) -> str:
    path = os.path.join(
        os.path.dirname(temp_path),
        os.path.basename(temp_path).split("temp_")[1]
    )
    return path


def copy_temp_file(
        output_dir: str,
        file_path: str
) -> str:
    output_path = get_temp_path(output_dir=output_dir, file_path=file_path)
    logger.debug(f"Copying from {file_path} to {output_path}")
    shutil.copyfile(file_path, output_path)
    return output_path


def parse_image_list(
        sub_dir: str | int = "",
        group_by_object: bool = True,
        base_dir_f: Callable[[str], str] = raw_img_dir
):
    
    object_dict = dict()
        
    img_list = glob(f'{base_dir_f(sub_dir)}/*.fits')

    if not group_by_object:
        return sorted(img_list)
    
    for img_file in img_list:
        img = fits.open(img_file)
        
        obj = img[0].header['OBJECT']
        
        if obj not in object_dict.keys():
            object_dict[obj] = [img_file]
        else:
            object_dict[obj].append(img_file)
            
    logger.debug(f'Data contains {len(object_dict.keys())} objects: {list(object_dict.keys())}')
    
    for key in ["dark", "flat", "bias"]:
        if key not in object_dict.keys():
            object_dict[key] = []

    return object_dict


raw_img_key = "RAWPATH"
base_name_key = "BASENAME"
proc_history_key = "CALSTEPS"
latest_save_key = "SAVEPATH"
latest_mask_save_key = "MASKPATH"
saturate_key = "SATURATE"
sextractor_header_key = 'SRCCAT'
psfex_header_key = 'PSFCAT'
norm_psfex_header_key = 'NPSFCAT'
flat_frame_key = 'FLATNAME'
bias_frame_key = 'BIASNAME'
dark_frame_key = 'DARKNAME'
coadd_key = "COADDS"

core_fields = ["OBSCLASS", "TARGET", "UTCTIME", coadd_key, proc_history_key]


class ProcessingError(Exception):
    pass

