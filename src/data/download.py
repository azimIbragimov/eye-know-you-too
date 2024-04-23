# This work is licensed under a "Creative Commons Attribution-NonCommercial-
# ShareAlike 4.0 International License"
# (https://creativecommons.org/licenses/by-nc-sa/4.0/).
#
# Author: Dillon Lohr (djl70@txstate.edu)
# Property of Texas State University.

import functools
import shutil
from pathlib import Path
from typing import Union

import requests
from tqdm.auto import tqdm


def download(url: str, save_to: Union[str, Path]) -> Path:
    """
    Download and save a file from a URL.

    Parameters
    ----------
    url : str
        The URL containing the file to download.
    save_to : str or Path
        The path to which the file will be saved.

    Returns
    -------
    out_path : Path
        The path to which the file was saved.

    References
    ----------
    https://stackoverflow.com/a/63831344
    """
    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()
        raise RuntimeError(
            f"Request to {url} returned status code {r.status_code}"
        )
    file_size = int(r.headers.get("Content-Length", 0))

    out_path = Path(save_to).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    desc = "(Unknown total file size)" if file_size == 0 else ""
    r.raw.read = functools.partial(r.raw.read, decode_content=True)
    with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
        with out_path.open("wb") as f:
            shutil.copyfileobj(r_raw, f)

    return out_path
