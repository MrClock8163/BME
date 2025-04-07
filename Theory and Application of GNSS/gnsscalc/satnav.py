import os
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd
import georinex as gr

from . import download_brdc, extract_brdc


GPST_START = datetime(1980, 1, 6)
GPST_LEAP = 18
GST_START = GPST_START
GST_LEAP = GPST_LEAP
BDT_START = datetime(2006, 1, 1)
BDT_LEAP = 4
WEEKSEC = 7 * 24 * 3600


class Orbit:
    def __init__(
        self,
        epoch: datetime,
        directory: str
    ):
        self.epoch: datetime = epoch
        if not os.path.isdir(directory):
            raise ValueError("specified path does not exist or is not a directory")

        gz = download_brdc(self.epoch, directory)
        self.brdcpath: str = extract_brdc(gz)
        self.satellites: set[str] = set()
        self.navdata: pd.DataFrame | None = None
        self.sat_xyz: pd.DataFrame | None = None

    def epoch_to_gps_gal_bd_systimes(
        self
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """Converts UTC epoch into system time week-seconds pairs
        for the GPS, Galileo and Beidou constellations.

        :return: system times
        :rtype: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
        """
        gpst_elapsed = self.epoch - GPST_START + timedelta(seconds=GPST_LEAP)
        gst_elapsed = gpst_elapsed
        bdt_elapsed = self.epoch - BDT_START + timedelta(seconds=BDT_LEAP)

        return (
            divmod(gpst_elapsed.total_seconds(), WEEKSEC),
            divmod(gst_elapsed.total_seconds(), WEEKSEC),
            divmod(bdt_elapsed.total_seconds(), WEEKSEC)
        )

    @staticmethod
    def solve_kepler(mean_anomaly: np.ndarray, e: np.ndarray, tol=1e-9) -> np.ndarray:
        num_sat = mean_anomaly.size
        diff = np.ones((num_sat, 1)) * np.inf
        e0 = mean_anomaly + e * np.sin(mean_anomaly)
        while any(abs(diff) > tol):
            e1 = mean_anomaly + e * np.sin(e0)
            diff = e1 - e0
            e0 = e1

        return e0

    @staticmethod
    def solve_ode(state: np.ndarray, ls_acc: np.ndarray) -> np.ndarray:
        """Returns the vector of derivatives by computing the system of ODEs.

            :param state: 6-element array containing [x, y, z, dx, dy, dz].
            :type state: np.ndarray
            :param ls_acc: 3-element array containig the lunisolar accelerations:
                `[x_ls, y_ls, z_ls]`.
            :type ls_acc: np.ndarray
            :return: 6-element array containing the derivatives
                [dx, dy, dz, ddx, ddy, ddz].
            :rtype: np.ndarray
        """

        MU = 3.9860044e14
        C20 = -1.08263e-3
        ELL_A = 6378136
        OMEGA = 7.292115e-5

        x = state[0]
        y = state[1]
        z = state[2]
        dx = state[3]
        dy = state[4]
        dz = state[5]
        xls = ls_acc[0]
        yls = ls_acc[1]
        zls = ls_acc[2]

        r = np.sqrt(x**2 + y**2 + z**2)
        ddx = (
            -MU / pow(r, 3) * x
            + 3 / 2 * C20 * MU * pow(ELL_A, 2) / pow(r, 5) * x * (1 - 5 * pow(z, 2) / pow(r, 2))
            + xls + pow(OMEGA, 2) * x + 2 * OMEGA * dy
        )
        ddy = (
            -MU / pow(r, 3) * y
            + 3 / 2 * C20 * MU * pow(ELL_A, 2) / pow(r, 5) * y * (1 - 5 * pow(z, 2) / pow(r, 2))
            + yls + pow(OMEGA, 2) * y - 2 * OMEGA * dx
        )
        ddz = (
            -MU / pow(r, 3) * z
            + 3 / 2 * C20 * MU * pow(ELL_A, 2) / pow(r, 5) * z * (3 - 5 * pow(z, 2) / pow(r, 2))
            + zls
        )

        return np.array([dx, dy, dz, ddx, ddy, ddz])

    def read_usable_navdata(
        self,
        constellation: Literal['G', 'E', 'C']
    ) -> pd.DataFrame | None:
        """Reads usable navigation data for satellites in GPS, Galileo or BeiDou constellation.

        :param constellation: satellite constellation to load (`G`: GPS, `E`: Galileo, `C`: BeiDou)
        :type constellation: Literal['G', 'E', 'C']
        :raises ValueError: unknown constellation
        :return: navigation data in multiindex dataframe
        :rtype: pandas.DataFrame | None
        """
        systimes = self.epoch_to_gps_gal_bd_systimes()

        brdc = gr.load(self.brdcpath, use=constellation).to_dataframe().dropna(how="all")
        match constellation:
            case 'G':
                health = "health"
                week, sec = systimes[0]
            case 'E':
                health = "health"
                week, sec = systimes[1]
            case 'C':
                health = "SatH1"
                week, sec = systimes[2]
            case _:
                raise ValueError(f"Unknown constellation: {constellation}")

        healthy = brdc[brdc[health] == 0]
        satellites = [
            idx for idx in healthy.index.get_level_values("sv").unique().tolist() if len(idx) == 3
        ]
        self.satellites.update(satellites)

        data: list[pd.DataFrame] = []
        for sat in satellites:
            sat_data = healthy.xs(sat, level="sv", drop_level=False)

            match constellation:
                case 'G':
                    elapsed = abs((week - sat_data.GPSWeek) * WEEKSEC + (sec - sat_data.Toe))
                case 'E':
                    elapsed = abs((week - sat_data.GALWeek) * WEEKSEC + (sec - sat_data.Toe))
                case 'C':
                    elapsed = abs((week - sat_data.BDTWeek) * WEEKSEC + (sec - sat_data.Toe))
                case _:
                    raise ValueError(f"Unknown constellation: {constellation}")

            elapsed = elapsed[elapsed <= 4 * 3600]
            if len(elapsed) == 0:
                continue
            min_idx = elapsed.index[elapsed.argmin()]

            data.append(sat_data.loc[[min_idx]])

        if len(data) == 0:
            return None

        return pd.concat(data, axis=0)

    def read_usable_navdata_glonass(
        self
    ) -> pd.DataFrame | None:
        """Reads usable navigation data for GLONASS satellites.

        :return: navigation data in multiindex dataframe
        :rtype: pandas.DataFrame | None
        """
        systime = np.datetime64(self.epoch)
        brdc = gr.load(self.brdcpath, use='R').to_dataframe().dropna(how="all")
        healthy = brdc[brdc["health"] == 0]
        satellites = [
            idx for idx in healthy.index.get_level_values("sv").unique().tolist() if len(idx) == 3
        ]
        self.satellites.update(satellites)

        data: list[pd.DataFrame] = []
        for sat in satellites:
            sat_data = healthy.xs(sat, level="sv", drop_level=False)
            elapsed = abs(systime - sat_data.index.get_level_values("time"))
            idx = sat_data.index.values[elapsed <= np.timedelta64(15 * 60, 's')]
            if len(idx) == 0:
                continue

            data.append(sat_data.loc[[idx[0]]])

        if len(data) == 0:
            return None

        return pd.concat(data, axis=0)

    def load_usable_navdata(self):
        """Loads usable navigation data for all constellations.

        :raises Exception: no usable data was found for any constellation
        """
        gps_data = self.read_usable_navdata('G')
        gal_data = self.read_usable_navdata('E')
        bd_data = self.read_usable_navdata('C')
        glo_data = self.read_usable_navdata_glonass()
        valid_data = [
            data for data in [gps_data, gal_data, bd_data, glo_data] if data is not None
        ]
        if len(valid_data) == 0:
            raise Exception("No usable navigation data was found")

        self.navdata = pd.concat(valid_data, axis=0).reset_index(level="time")

    def calc_ecef_xyz_gps_gal_bd(self) -> pd.DataFrame:
        """Caculates ECEF XYZ coordinates of GPS, Galileo and BeiDou satellites
        at the observation epoch.

        :return: ECEF XYZ coordinates in dataframe
        :rtype: pandas.DataFrame
        """
        sats_gps = list(filter(lambda n: n[0] == "G", self.satellites))
        sats_gal = list(filter(lambda n: n[0] == "E", self.satellites))
        sats_bd = list(filter(lambda n: n[0] == "C", self.satellites))

        data: pd.DataFrame = self.navdata.loc[sats_gps + sats_gal + sats_bd]

        n_gps = len(sats_gps)
        n_gal = len(sats_gal)
        n_bd = len(sats_bd)

        gpst, gst, bdt = self.epoch_to_gps_gal_bd_systimes()
        gpst_obs = gst_obs = gpst[0] * WEEKSEC + gpst[1]
        bdt_obs = bdt[0] * WEEKSEC + bdt[1]

        t_obs_sys = np.array(
            ([gpst_obs] * n_gps) + ([gst_obs] * n_gal) + ([bdt_obs] * n_bd)
        )

        weeks = pd.concat(
            [
                data.loc[sats_gps].GPSWeek,
                data.loc[sats_gal].GALWeek,
                data.loc[sats_bd].BDTWeek
            ]
        )
        t_oe = weeks * WEEKSEC + data.Toe
        t_c = t_obs_sys - t_oe

        dt_sv = (
            data.SVclockBias
            + data.SVclockDrift * t_c
            + data.SVclockDriftRate * pow(t_c, 2)
        )
        t_oe_sys0 = t_oe - dt_sv
        t_k0 = t_obs_sys - t_oe_sys0

        mu = np.array(
            ([3.986005e+14] * n_gps)
            + ([3.986004418e+14] * n_gal)
            + ([3.986004418e+14] * n_bd)
        )
        omega_e = np.array(
            ([7.2921151467e-5] * n_gps)
            + ([7.2921151467e-5] * n_gal)
            + ([7.2921150e-5] * n_bd)
        )

        a = pow(data.sqrtA, 2)
        n0 = np.sqrt(mu / pow(a, 3))
        n = n0 + data.DeltaN
        ecc = data.Eccentricity
        m_k0 = data.M0 + n * t_k0
        e_k0 = self.solve_kepler(m_k0, ecc)

        c = 2.99792458e+8
        f = (-2 * np.sqrt(mu)) / pow(c, 2)
        dt_rel = f * ecc * data.sqrtA * np.sin(e_k0)

        dt_sv += dt_rel
        t_oe_sys = t_oe - dt_sv
        t_k = t_obs_sys - t_oe_sys
        m_k = data.M0 + n * t_k
        e_k = self.solve_kepler(m_k, ecc)  # MISTAKE IN HANDOUT HTML!

        # Real anomaly.
        nu_k = 2 * np.arctan(np.sqrt(1 + ecc) / np.sqrt(1 - ecc) * np.tan(e_k / 2))

        u = nu_k + data.omega
        du = data.Cus * np.sin(2 * u) + data.Cuc * np.cos(2 * u)  # Argument of latitude
        dr = data.Crs * np.sin(2 * u) + data.Crc * np.cos(2 * u)  # Radial distance
        di = data.Cis * np.sin(2 * u) + data.Cic * np.cos(2 * u)  # Inclination

        # Final parameter values.
        u_k = u + du
        r_k = a * (1 - ecc * np.cos(e_k)) + dr
        i_k = data.Io + data.IDOT * t_k + di

        xp_k = r_k * np.cos(u_k)
        yp_k = r_k * np.sin(u_k)
        coord_orb = pd.DataFrame({
            "x_k": xp_k,
            "y_k": yp_k,
            "z_k": np.zeros(len(xp_k))
        })

        sow_sys = t_oe_sys % WEEKSEC  # Seconds inside the week.
        l_k = data.Omega0 + (data.OmegaDot - omega_e) * t_k - omega_e * sow_sys

        coord_ecef: list[np.ndarray] = []
        for inc, lon, x_orb in zip(i_k, l_k, coord_orb.to_numpy()):
            # Rotation around the x'k by the inclination.
            R1 = np.array([
                [1,         0,          0],
                [0, np.cos(inc), -np.sin(inc)],
                [0, np.sin(inc),  np.cos(inc)]
            ])
            R2 = np.array([
                [np.cos(lon), -np.sin(lon), 0],
                [np.sin(lon),  np.cos(lon), 0],
                [0,          0, 1]
            ])
            R_ecef = R2 @ R1
            coord_ecef.append(R_ecef @ x_orb)

        df = pd.DataFrame(
            coord_ecef,
            columns=["x", "y", "z"],
            index=data.index.to_list()
        )
        df.index.name = "sv"

        return df

    def calc_ecef_xyz_glo(self) -> pd.DataFrame:
        """Calculates ECEF XYZ coordinates of GLONASS satellites
        at the observation epoch.

        :return: ECEF XYZ coordinates in dataframe
        :rtype: pandas.DataFrame
        """
        t_obs_utc = np.datetime64(self.epoch)
        satellites = [idx for idx in self.satellites if idx[0] == "R"]
        data: pd.DataFrame = self.navdata.loc[satellites]

        tau_n = data.SVclockBias
        tau_n_delta = pd.to_timedelta(tau_n, 's')
        t_oe_utc = data.time - tau_n_delta
        t_k = (t_obs_utc - t_oe_utc) / np.timedelta64(1, 's')

        interval = t_k
        stepsize = 10
        step_spec: pd.DataFrame = pd.concat(divmod(interval, stepsize), axis=1)
        step_spec.columns = ["full", "partial"]

        coord_ecef: list[np.ndarray] = []
        for (sv, sv_data), (_, step_data) in zip(data.iterrows(), step_spec.iterrows()):
            init_state = sv_data[["X", "Y", "Z", "dX", "dY", "dZ"]].to_numpy().flatten()
            ls_acc = sv_data[["dX2", "dY2", "dZ2"]].to_numpy().flatten()
            full, partial = step_data.to_numpy().flatten()
            step_vec = np.ones(int(abs(full))) * stepsize * np.sign(full)
            step_vec = np.append(step_vec, partial)

            y0 = init_state

            for step in step_vec:
                # Point A
                m1 = self.solve_ode(y0, ls_acc)
                ya = y0 + m1 * step/2

                # Point B
                m2 = self.solve_ode(ya, ls_acc)
                yb = y0 + m2 * step/2

                # Point C
                m3 = self.solve_ode(yb, ls_acc)
                yc = y0 + m3 * step

                # Point D
                m4 = self.solve_ode(yc, ls_acc)

                new_state = y0 + 1/6*(m1 + 2*m2 + 2*m3 + m4)*step
                y0 = new_state

            coord_ecef.append(y0[:3])

        df = pd.DataFrame(
            coord_ecef,
            columns=["x", "y", "z"],
            index=data.index.to_list()
        )
        df.index.name = "sv"
        return df

    def calc_sat_ecef_xyz(self):
        """Calculates ECEF XYZ coordinates of all satellites
        at the observation epoch and stores them internally.
        """
        ecef = self.calc_ecef_xyz_gps_gal_bd()
        ecef_glo = self.calc_ecef_xyz_glo()
        self.sat_xyz = pd.concat([ecef, ecef_glo], axis=0)
