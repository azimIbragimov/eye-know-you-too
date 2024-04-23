# This work is licensed under a "Creative Commons Attribution-NonCommercial-
# ShareAlike 4.0 International License"
# (https://creativecommons.org/licenses/by-nc-sa/4.0/).
#
# Author: Dillon Lohr (djl70@txstate.edu)
# Property of Texas State University.

import tarfile
import zipfile
from pathlib import Path


def extract(archive: Path, to: Path) -> None:
    suffix_to_func = {".zip": unzip, ".gz": untar}
    suffix = archive.suffix
    if suffix not in suffix_to_func:
        raise RuntimeError("Unsupported archive format:", suffix)
    suffix_to_func[suffix](archive, to)


def unzip(archive: Path, to: Path) -> None:
    """
    Unzip an archive to the given directory.

    Parameters
    ----------
    archive : Path
        The path to the archive that will be unzipped.
    to : Path
        The directory to which the archive will be unzipped.
    """
    with zipfile.ZipFile(archive, "r") as zip_ref:
        zip_ref.extractall(to)


def untar(archive: Path, to: Path) -> None:
    """
    Untar an archive to the given directory.

    Parameters
    ----------
    archive : Path
        The path to the archive that will be untarred.
    to : Path
        The directory to which the archive will be untarred.
    """
    with tarfile.open(archive, "r:gz") as tar_ref:
        tar_ref.extractall(to)
