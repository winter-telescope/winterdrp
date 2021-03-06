import os
import logging
from winterdrp.paths import raw_img_dir


logger = logging.getLogger(__name__)


def download_via_ssh(
        server: str,
        base_dir: str,
        night: str | int,
        pipeline: str,
        server_sub_dir: str = None
):
    username = input(f"Please enter your username for {server}: \n")

    source_dir = f"{username}@{server}:{os.path.join(base_dir, night)}/"

    if server_sub_dir is not None:
        source_dir += f"{server_sub_dir}/"

    output_dir = raw_img_dir(os.path.join(pipeline, night))

    try:
        os.makedirs(output_dir)
    except OSError:
        pass

    cmd = f"rsync -a -v --include '*.fits' --exclude '*' {source_dir} {output_dir}"

    logger.info(f"Executing '{cmd}'")

    os.system(cmd)

