import os
import logging
import subprocess
import numpy as np
from pathlib import Path
from winterdrp.paths import calibration_config_dir
from winterdrp.utils.dockerutil import new_container, docker_path, docker_put, docker_get_new_files

logger = logging.getLogger(__name__)

sextractor_cmd = os.getenv("SEXTRACTOR_CMD")


def local_sextractor(
        cmd: str,
        output_dir: str
):
    """
    Function to run sextractor on local machine using subprocess. It will only work if you have installed sextractor
    correctly, and specified the command to run sextractor with:
        export SEXTRACTOR_CMD = '/path/to/sextractor/executable/file

    After sextractor has been run using the specified 'cmd' command,
     all newly-generated files will be copied out of the container to 'output_dir'

    Parameters
    ----------
    cmd: A string containing the command you want to use to run sextractor. An example would be:
        cmd = '/usr/bin/source-extractor image0001.fits -c sex.config'
    output_dir: A local directory to save the output files to.

    Returns
    -------

    """

    try:
        rval = subprocess.run(cmd.split(" "), check=True, capture_output=True)
        rval = subprocess.run(cmd.split(), check=True, capture_output=True)
        logger.info('Process completed')
        logger.debug(rval.stdout.decode())

        raise NotImplementedError

        return 0

    except subprocess.CalledProcessError as err:
        logger.error(f'Could not run sextractor with error {err}')
        return -1


def docker_sextractor(
        cmd: str,
        output_dir: str,
):
    """Function to run sextractor via Docker. A container will be generated automatically,
    but a Docker server must be running first. You can start one via the Desktop application,
    or on the command line with `docker start'.

    After sextractor has been run using the specified 'cmd' command,
     all newly-generated files will be copied out of the container to 'output_dir'

    Parameters
    ----------
    cmd: A string containing the base arguments you want to use to run sextractor. An example would be:
        cmd = 'image01.fits -c sex.config'
    output_dir: A local directory to save the output files to.

    Returns
    -------

    """

    container = new_container()
    container.attach()

    container.start()

    split = cmd.split(" -")

    new_split = []

    # Loop over sextractor command, and
    # copy everything that looks like a file into container

    copied = []

    for i, arg in enumerate(split):
        sep = arg.split(" ")
        if os.path.exists(sep[1]):
            new = list(sep)
            new[1] = docker_path(sep[1])
            new_split.append(" ".join(new))
            docker_put(container, sep[1])
            copied.append(sep[1])
        else:
            new_split.append(arg)

    cmd = " -".join(new_split)

    # See what files are already there

    ignore_files = container.exec_run("ls", stderr=True, stdout=True).output.decode().split("\n")

    # Run sextractor

    log = container.exec_run(cmd, stderr=True, stdout=True)

    if not log.output == b"":
        logger.warning(f"Sextractor warning: {log.output.decode()}")

    docker_get_new_files(
        container=container,
        output_dir=output_dir,
        ignore_files=ignore_files
    )

    container.kill()
    container.remove()


# Either run sextractor locally or on docker

if sextractor_cmd is None:
    sextractor_cmd = "/usr/bin/source-extractor"
    execute_sextractor = docker_sextractor
else:
    execute_sextractor = local_sextractor


# Functions to parse commands and generate appropriate sextractor files

def parse_checkimage(
    checkimage_type: str | list = None,
    image: str = None,
):
    """Function to parse the "checkimage" component of Sextractor configuration.

    Parameters
    ----------
    checkimage_type: The 'CHECKIMAGE_TYPE' files for sextractor. The default is None. To quote sextractor,
    available types are: 'NONE, BACKGROUND, BACKGROUND_RMS, MINIBACKGROUND, MINIBACK_RMS, -BACKGROUND,
    FILTERED, OBJECTS, -OBJECTS, SEGMENTATION, or APERTURES'. Multiple arguments should be specified in a list.
    image: The name of the image in question. If specified, the name of each checkimage will include the
    name of the original base image.

    Returns
    -------
    cmd: A string containing the partial sextractor command relating to checkimages. The default is an empty string.
    """
    if isinstance(checkimage_type, str):
        checkimage_type = list(checkimage_type)

    cmd = ""

    if image is not None:
        base_name = f'{os.path.basename(image).split(".")[0]}_'
    else:
        base_name = ""

    if checkimage_type is not None:
        cmd = "-CHECKIMAGE_TYPE " + ",".join(checkimage_type)
        cmd += " -CHECKIMAGE_NAME " + ",".join([
            f"{base_name}check_{x.lower()}.fits" for x in checkimage_type
        ])
        cmd += " "

    return cmd


def run_sextractor(
        images: str | list,
        output_dir: str,
        config: str = os.path.join(calibration_config_dir, 'astrom.sex'),
        param: str = os.path.join(calibration_config_dir, 'astrom.param'),
        filter_name: str = os.path.join(calibration_config_dir, 'default.conv'),
        star_nnw: str = os.path.join(calibration_config_dir, 'default.nnw'),
        weight_image: str = None,
        verbose_type: str = "QUIET",
        checkimage_type: str | list = None,
        reprocess: bool = True
):

    if not isinstance(images, list):
        images = [images]

    for img in images:
        image_name = Path(img).stem
        output_catalog = f'{image_name}.cat'

        cmd = f"{sextractor_cmd} {img} " \
              f"-c {config} " \
              f"-CATALOG_NAME {output_catalog} " \
              f"-PARAMETERS_NAME {param} " \
              f"-FILTER_NAME {filter_name} " \
              f"-STARNNW_NAME {star_nnw} " \
              f"-VERBOSE_TYPE {verbose_type} "

        cmd += parse_checkimage(
            checkimage_type=checkimage_type,
            image=img
        )

        if weight_image is None:
            cmd += "-WEIGHT_TYPE None"
        else:
            cmd += f"-WEIGHT_IMAGE {weight_image}"

        if not reprocess:

            output_cat_path = os.path.join(output_dir, output_catalog)

            if os.path.exists(output_cat_path):
                logger.debug(f"Skipping because {output_cat_path} already exist.")
                continue

        logger.debug(f"Using '{['local', 'docker'][sextractor_cmd == local_sextractor]}' "
                     f"sextractor installation to run `{cmd}`")

        execute_sextractor(cmd, output_dir)


if __name__ == "__main__":
    run_sextractor(
        "/Users/robertstein/Data/WIRC/20200929/redux/image0240.fits",
        "/Users/robertstein/Data/testersextractor",
        checkimage_type=["BACKGROUND", "BACKGROUND_RMS"]
    )
