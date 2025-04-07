from __future__ import annotations

from math import sqrt

import pandas as pd
import numpy as np


class Ellipsoid:
    """Ellipsoid parameter definition."""
    def __init__(self, a: float, f: float):
        """Creates ellipsoid definiton from semi-major and flattening.

        :param a: semi-major axis
        :type a: float
        :param b: flattening
        :type b: float
        :return: ellipsoid definiton
        :rtype: Ellipsoid
        """
        self.a: float = a
        """Semi-major axis"""

        self.f: float = f
        """Flattening"""

        self.invf: float = 1 / f
        """Inverse flattening"""

        self.b: float = self.a - self.a * self.f
        """Semi-minor axis"""

        self.e1: float = sqrt((self.a**2 - self.b**2) / self.a**2)
        """First numberical eccentricity"""

        self.e12: float = self.e1**2
        """Square of first numerical eccentricity"""

        self.e2: float = sqrt((self.a**2 - self.b**2) / self.b**2)
        """Second numberical eccentricity"""

        self.e22: float = self.e2**2
        """Square of second numerical eccentricity"""

        self.c = a * self.e1
        """Linear eccentricity"""
    
    def __repr__(self) -> str:
        return (
            f"a: {self.a}, b: {self.b}, f: {self.f}, "
            f"invf: {self.invf}  e1: {self.e1}, e2: {self.e2}, c: {self.c}"
        )

    @classmethod
    def from_a_b(cls, a: float, b: float) -> Ellipsoid:
        """Creates ellipsoid definiton from semi-major and minor axes.

        :param a: semi-major axis
        :type a: float
        :param b: semi-minor axis
        :type b: float
        :return: ellipsoid definiton
        :rtype: Ellipsoid
        """
        f = (a - b) / a
        return cls(a, f)

    @classmethod
    def from_a_invf(cls, a: float, invf: float) -> Ellipsoid:
        """Creates ellipsoid definition from semi-major
        axis and inverse flattening.

        :param a: semi-major axis
        :type a: float
        :param invf: inverse flattening
        :type invf: float
        :return: ellipsoid definition
        :rtype: Ellipsoid
        """
        return cls(a, 1 / invf)

    @classmethod
    def from_a_e1(cls, a: float, e1: float) -> Ellipsoid:
        """Creates ellipsoid definition from semi-major axis
        and first numerical eccentricity.

        :param a: semi-major axis
        :type a: float
        :param e1: first numerical eccentricity
        :type e1: float
        :return: ellipsoid definition
        :rtype: Ellipsoid
        """
        f = 1 - sqrt(1 - e1**2)
        return cls(a, f)
    
    @classmethod
    def wgs84(cls) -> Ellipsoid:
        """Creates WGS84 ellipsoid definition.

        :return: WGS84 ellpisoid
        :rtype: Ellipsoid
        """
        return cls.from_a_invf(6_378_137, 298.257_223_563)


class Transformer:
    def __init__(self, ellipsoid: Ellipsoid):
        self.ell: Ellipsoid = ellipsoid
    
    def xyz2llh(self, coo: pd.DataFrame) -> pd.DataFrame:
        """Transforms cartesian coordinate to ellipsoidal coordinates.

        :param coo: `x-y-z` data
        :type coo: pandas.DataFrame
        :return: transformed ellipsoidal coordinates in `lat-lon-h` frame
        :rtype: pandas.DataFrame
        """
        x = coo.x
        y = coo.y
        z = coo.z

        lamb = np.degrees(np.atan2(y, x))

        p = np.sqrt(x**2 + y**2)
        phi_0 = np.atan(z / ((1 - self.ell.e12) * p))
        tol = 1e-9
        diff = np.full(x.shape, np.inf)

        while (diff >= tol).any():
            N_i = self.ell.a / np.sqrt(1 - self.ell.e12 * np.sin(phi_0)**2)
            h_i = p / np.cos(phi_0) - N_i
            phi_i = np.atan(z / (p * (1 - self.ell.e12 * N_i / (N_i + h_i))))

            diff = phi_i - phi_0
            phi_0 = phi_i

        N_i = self.ell.a / np.sqrt(1 - self.ell.e12 * np.sin(phi_0)**2)
        h_i = p / np.cos(phi_0) - N_i
        phi_i = np.atan(z / (p * (1 - self.ell.e12 * N_i / (N_i + h_i))))
        h_i = p / np.cos(phi_0) - N_i

        phi = np.degrees(phi_i)

        df = pd.concat(
            [phi, lamb, h_i],
            axis=1
        )
        df.columns = ["lat", "lon", "h"]

        return df
    
    def llh2xyz(self, coo: pd.DataFrame) -> pd.DataFrame:
        """Transforms ellipsoidal coordinates to cartesian coordinates.

        :param coo: `lat-lon-h` data
        :type coo: pandas.DataFrame
        :return: transformed cartesian coordinates in `x-y-z` frame
        :rtype: pandas.DataFrame
        """
        llh = coo.iloc[:,1:].to_numpy()
        lat = np.radians(llh[:, 0])
        lon = np.radians(llh[:, 1])
        h = llh[:, 2]

        N = self.ell.a / np.sqrt(1 - self.ell.e12 * np.sin(lat)**2)

        p = (N + h) * np.cos(lat)

        x = p * np.cos(lon)
        y = p * np.sin(lon)
        z = ((1 - self.ell.e12) * N + h) * np.sin(lat)

        df: pd.DataFrame = pd.concat(
            [x, y, z],
            axis=1
        )
        df.columns = ["x", "y", "z"]

        return df
    
    def xyz2aer(self, coo: pd.DataFrame, center: pd.DataFrame) -> pd.DataFrame:
        """Transforms cartesian coordinates to topocentric
        and horizontal coordinates at the given center location.

        :param coo: `x-y-z` data
        :type coo: pandas.DataFrame
        :param center: ˙x-y-z` data of the center
        :type center: pandas.DataFrame
        :return: transformed topocentric horizontal coordinates
            in `east-north-up-azimut-elevation-range` frame
        :rtype: pandas.DataFrame
        """
        xyz = coo
        rec = center

        rec_llh = self.xyz2llh(center)
        lat, lon, h = np.radians(rec_llh).to_numpy().flatten()

        offset = xyz - rec.to_numpy()

        rot_topo = np.array(
            [
                [-np.sin(lon),              np.cos(lon),             0],
                [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
                [ np.cos(lat)*np.cos(lon),  np.cos(lat)*np.sin(lon), np.sin(lat)]
            ]
        )
        
        topo = offset @ rot_topo.T

        east = topo[0]
        north = topo[1]
        up = topo[2]

        a = np.degrees(np.atan2(east, north)) % 360.0
        e = np.degrees(np.atan2(up, np.sqrt(east**2 + north**2)))
        r = np.linalg.norm(offset,axis=1)

        df: pd.DataFrame = pd.concat(
            [east, north, up, a, e],
            axis=1
        )
        df.columns = ["easting", "northing", "up", "azimut", "elevation"]
        df["range"] = r

        return df
