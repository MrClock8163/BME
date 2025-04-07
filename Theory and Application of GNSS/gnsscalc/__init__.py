from datetime import datetime
from urllib.request import urlretrieve
import os
from typing import Literal

import gzip

import pandas as pd


def download_brdc(date: datetime, dirpath: str) -> str:
    """Downloads RINEX navigation file from BKG repository
    for the given date.

    :param date: observation date
    :type date: datetime
    :param dirpath: directory to save the downloaded file
    :type dirpath: str
    :return: path to the downloaded file
    :rtype: str
    """
    doy = date.timetuple().tm_yday
    filepath = os.path.join(dirpath, f"BRDM00DLR_S_{date.year:d}{doy:03d}0000_01D_MN.rnx.gz")
    if os.path.isfile(filepath):
        return filepath

    url = f"https://igs.bkg.bund.de/root_ftp/IGS/BRDC/{date.year:d}/{doy:03d}/BRDM00DLR_S_{date.year:d}{doy:03d}0000_01D_MN.rnx.gz"
    path, response = urlretrieve(url, filepath)

    return path


def extract_gzip(filepath: str) -> str:
    """Extracts compressed RINEX navigation file.

    :param filepath: path to the compressed file
    :type filepath: str
    :return: path to the decompressed file
    :rtype: str
    """
    outpath = os.path.splitext(filepath)[0]
    if os.path.isfile(outpath):
        return outpath

    with gzip.open(filepath, "rb") as zip, open(outpath, "wb") as f:
        f.write(zip.read())

    return outpath


def load_coords(coo: list[tuple[str, float, float, float]], format: Literal['XYZ', 'LLH']) -> pd.DataFrame:
    """Loads coordinate list into appropriate data frame.

    :param coo: 2D list of coordinates as `[..., [id, v1, v2, v3], ...]`
    :type coo: list[list[str, float, float, float]]
    :param format: coordinate format
    :type format: Literal['XYZ', 'LLH']
    :raises ValueError: exception is raised if an unknown coordinate format was passed
    :return: data frame with appropriate column names and data types
    :rtype: pandas.DataFrame
    """
    match format:
        case 'LLH':
            colnames = ("lat", "lon", "h")
        case 'XYZ':
            colnames = ("x", "y", "z")
        case _:
            raise ValueError(f"Unknown coordinate format: {format}")

    colid: list[str] = []
    col1: list[float] = []
    col2: list[float] = []
    col3: list[float] = []
    for name, lat, lon, h in coo:
        colid.append(name)
        col1.append(lat)
        col2.append(lon)
        col3.append(h)

    df = pd.DataFrame(
        {
            "id": colid,
            colnames[0]: col1,
            colnames[1]: col2,
            colnames[2]: col3
        }
    ).astype(
        {
            "id": "string",
            colnames[0]: "float64",
            colnames[1]: "float64",
            colnames[2]: "float64"
        }
    ).set_index("id")

    return df
