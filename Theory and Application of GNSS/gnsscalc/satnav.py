import os
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd
import georinex as gr

from .geoid import GeoidHeight


GPST_START = datetime(1980, 1, 6)
GPST_LEAP = 18
GST_START = GPST_START
GST_LEAP = GPST_LEAP
BDT_START = datetime(2006, 1, 1)
BDT_LEAP = 4
WEEKSEC = 7 * 24 * 3600


def epoch_to_gps_gal_bd_systimes(
    epoch: datetime
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Converts UTC epoch into system time week-seconds pairs
    for the GPS, Galileo and Beidou constellations.

    :return: system times
    :rtype: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    """
    gpst_elapsed = epoch - GPST_START + timedelta(seconds=GPST_LEAP)
    gst_elapsed = gpst_elapsed
    bdt_elapsed = epoch - BDT_START + timedelta(seconds=BDT_LEAP)

    return (
        divmod(gpst_elapsed.total_seconds(), WEEKSEC),
        divmod(gst_elapsed.total_seconds(), WEEKSEC),
        divmod(bdt_elapsed.total_seconds(), WEEKSEC)
    )


def read_usable_navdata(
    brdcpath: str,
    epoch: datetime,
    constellation: Literal['G', 'E', 'C']
) -> pd.DataFrame | None:
    """Reads usable navigation data for satellites in GPS, Galileo or BeiDou constellation.

    :param constellation: satellite constellation to load (`G`: GPS, `E`: Galileo, `C`: BeiDou)
    :type constellation: Literal['G', 'E', 'C']
    :raises ValueError: unknown constellation
    :return: navigation data in multiindex dataframe
    :rtype: pandas.DataFrame | None
    """
    systimes = epoch_to_gps_gal_bd_systimes(epoch)

    brdc = gr.load(brdcpath, use=constellation).to_dataframe().dropna(how="all")
    match constellation:
        case 'G':
            health = "health"
            tgd = "TGD"
            week, sec = systimes[0]
        case 'E':
            health = "health"
            tgd = "BGDe5a"
            week, sec = systimes[1]
        case 'C':
            health = "SatH1"
            tgd = "TGD1"
            week, sec = systimes[2]
        case _:
            raise ValueError(f"Unknown constellation: {constellation}")

    healthy = brdc[brdc[health] == 0]
    satellites = [
        idx for idx in healthy.index.get_level_values("sv").unique().tolist() if len(idx) == 3
    ]
    # self.satellites.update(satellites)

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
    brdcpath: str,
    epoch: datetime
) -> pd.DataFrame | None:
    """Reads usable navigation data for GLONASS satellites.

    :return: navigation data in multiindex dataframe
    :rtype: pandas.DataFrame | None
    """
    systime = np.datetime64(epoch)
    brdc = gr.load(brdcpath, use='R').to_dataframe().dropna(how="all")
    healthy = brdc[brdc["health"] == 0]
    satellites = [
        idx for idx in healthy.index.get_level_values("sv").unique().tolist() if len(idx) == 3
    ]
    # self.satellites.update(satellites)

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


def load_usable_navdata(brdcpath: str, epoch: datetime):
    """Loads usable navigation data for all constellations.

    :raises Exception: no usable data was found for any constellation
    """
    gps_data = read_usable_navdata(brdcpath, epoch, 'G')
    gal_data = read_usable_navdata(brdcpath, epoch, 'E')
    bd_data = read_usable_navdata(brdcpath, epoch, 'C')
    glo_data = read_usable_navdata_glonass(brdcpath, epoch)
    valid_data = [
        data for data in [gps_data, gal_data, bd_data, glo_data] if data is not None
    ]
    if len(valid_data) == 0:
        raise Exception("No usable navigation data was found")

    return pd.concat(valid_data, axis=0).reset_index(level="time")


def solve_kepler(mean_anomaly: np.ndarray, e: np.ndarray, tol=1e-9) -> np.ndarray:
    num_sat = mean_anomaly.size
    diff = np.ones((num_sat, 1)) * np.inf
    e0 = mean_anomaly + e * np.sin(mean_anomaly)
    while any(abs(diff) > tol):
        e1 = mean_anomaly + e * np.sin(e0)
        diff = e1 - e0
        e0 = e1

    return e0

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



def calc_sat_ecef_gps_gal_bd(epochs: pd.Series, navdata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    def epochs_to_systimes(row: pd.Series) -> pd.Series:
        sv = row.loc["sv"]
        match sv[0]:
            case "G" | "E":
                row.loc[0] = (row.loc[0] - GPST_START + timedelta(seconds=GPST_LEAP)).total_seconds()
            case "C":
                row.loc[0] = (row.loc[0] - BDT_START + timedelta(seconds=BDT_LEAP)).total_seconds()
            case _:
                print(sv)
                raise Exception()

        return row

    sats = epochs.index.tolist()
    sats_gps = list(filter(lambda n: n[0] == "G" and n in navdata.index, sats))
    sats_gal = list(filter(lambda n: n[0] == "E" and n in navdata.index, sats))
    sats_bd = list(filter(lambda n: n[0] == "C" and n in navdata.index, sats))

    sats_relevant = sats_gps + sats_gal + sats_bd

    data: pd.DataFrame = navdata.loc[sats_relevant]
    epochs_relevant = epochs.loc[sats_gps + sats_gal + sats_bd] 

    n_gps = len(sats_gps)
    n_gal = len(sats_gal)
    n_bd = len(sats_bd)

    # gpst, gst, bdt = self.epoch_to_gps_gal_bd_systimes()
    # gpst_obs = gst_obs = gpst[0] * WEEKSEC + gpst[1]
    # bdt_obs = bdt[0] * WEEKSEC + bdt[1]

    # t_obs_sys = np.array(
    #     ([gpst_obs] * n_gps) + ([gst_obs] * n_gal) + ([bdt_obs] * n_bd)
    # )
    t_obs_sys = epochs_relevant.reset_index().apply(epochs_to_systimes, axis=1).set_index("sv", inplace=False)[0]
    # print(type(t_obs_sys), t_obs_sys)
    # exit()

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
    e_k0 = solve_kepler(m_k0, ecc)

    c = 2.99792458e+8
    f = (-2 * np.sqrt(mu)) / pow(c, 2)
    dt_rel = f * ecc * data.sqrtA * np.sin(e_k0)

    # dt_sv += dt_rel
    t_oe_sys = t_oe - dt_sv - dt_rel
    t_k = t_obs_sys - t_oe_sys
    m_k = data.M0 + n * t_k
    e_k = solve_kepler(m_k, ecc)  # MISTAKE IN HANDOUT HTML!

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
    df_aux = pd.concat(
        [
            pd.concat(
                [
                    data.loc[sats_gps].TGD,
                    data.loc[sats_gal].BGDe5a,
                    data.loc[sats_bd].TGD1
                ]
            ),
            dt_sv,
            dt_rel
        ],
        axis=1
    )
    df_aux.columns = ["tgd", "delta_s", "delta_rel"]

    return df, df_aux


def calc_sat_ecef_glo(
    epochs: pd.Series,
    navdata: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculates ECEF XYZ coordinates of GLONASS satellites
    at the observation epoch.

    :return: ECEF XYZ coordinates in dataframe
    :rtype: pandas.DataFrame
    """
    sats = epochs.index.tolist()
    satellites = [idx for idx in sats if idx[0] == "R"]
    data: pd.DataFrame = navdata.loc[satellites]

    tau_n = data.SVclockBias
    tau_n_delta = pd.to_timedelta(tau_n, 's')
    t_oe_utc = data.time - tau_n_delta
    t_k = (epochs - t_oe_utc).dropna() / np.timedelta64(1, 's')

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
            m1 = solve_ode(y0, ls_acc)
            ya = y0 + m1 * step/2

            # Point B
            m2 = solve_ode(ya, ls_acc)
            yb = y0 + m2 * step/2

            # Point C
            m3 = solve_ode(yb, ls_acc)
            yc = y0 + m3 * step

            # Point D
            m4 = solve_ode(yc, ls_acc)

            new_state = y0 + 1/6*(m1 + 2*m2 + 2*m3 + m4)*step
            y0 = new_state

        coord_ecef.append(y0[:3])

    df = pd.DataFrame(
        coord_ecef,
        columns=["x", "y", "z"],
        index=data.index.to_list()
    )
    df.index.name = "sv"
    df_aux = pd.concat(
        [   
            pd.Series(np.zeros([len(satellites)]), df.index),
            data.SVclockBias,
            pd.Series(np.zeros([len(satellites)]), df.index),
            data.FreqNum
        ],
        axis=1
    )
    df_aux.columns = ["tgd", "delta_s", "delta_rel", "freq_num"]

    return df, df_aux


def calc_sat_ecef(
    epochs: pd.Series,
    navdata: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gps_gal_bd, gps_gal_bd_aux = calc_sat_ecef_gps_gal_bd(epochs, navdata)
    # return gps_gal_bd
    glo, glo_aux = calc_sat_ecef_glo(epochs, navdata)

    return (
        pd.concat([gps_gal_bd, glo], axis=0),
        pd.concat([gps_gal_bd_aux, glo_aux], axis=0)
    )


def get_glo_frequency_numbers(navdata: pd.DataFrame) -> pd.DataFrame:
        """Returns the frequency number for all Glonass satellites.

        :return: Frequency numbers in dataframe
        :rtype: pandas.DataFrame
        """
        sats_glo = [sv for sv in navdata.index.to_list() if sv[0] == "R"]
        return navdata.FreqNum.loc[sats_glo].groupby(["sv"]).first()


def calc_corr_iono(brdcpath: str, epoch: datetime, navdata: pd.DataFrame, station_llh: pd.DataFrame, sats_aer: pd.DataFrame) -> pd.DataFrame:
    """Calculates slant ionospheric delays.

    :param station_llh: Observation station coordinates
    :type station_llh: pandas.DataFrame
    :param sats_aer: Horizontal coordinates of satellites
    :type sats_aer: pandas.DataFrame
    :return: Ionospheric corrections
    :rtype: pandas.DataFrame
    """
    sats_el_sc = sats_aer["elevation"] / 180
    sats_ipp_dist_sc = 0.0137 / (sats_el_sc + 0.11) - 0.022
    
    lat_sc, lon_sc, _ = station_llh.to_numpy().flatten() / 180
    ipp_lat_sc = lat_sc + sats_ipp_dist_sc * np.cos(np.radians(sats_aer["azimut"]))

    sc_75 = 75 / 180
    ipp_lat_sc.loc[ipp_lat_sc > sc_75] = sc_75

    ipp_lat = ipp_lat_sc * np.pi
    ipp_lon_sc = lon_sc + sats_ipp_dist_sc * np.sin(np.radians(sats_aer["azimut"])) / np.cos(ipp_lat)
    ipp_geolat_sc = ipp_lat_sc + 0.064 * np.cos((ipp_lon_sc - 1.617) * np.pi)

    epoch_sod = epoch.hour * 3600 + epoch.minute * 60 + epoch.second
    epoch_ipp = 43200 * ipp_lon_sc + epoch_sod
    
    header = gr.rinexheader(brdcpath)
    gpsa: list[float] = header["IONOSPHERIC CORR"]["GPSA"]
    gpsb: list[float] = header["IONOSPHERIC CORR"]["GPSB"]

    a2 = a4 = 0
    for i, (a, b) in enumerate(zip(gpsa, gpsb)):
        a2 += a * pow(ipp_geolat_sc, i)
        a4 += b * pow(ipp_geolat_sc, i)

    a2.loc[a2 < 0] = 0
    a4.loc[a4 < 72000] = 72000
    a1 = 5e-9
    a3 = 50400

    x = (2 * np.pi * (epoch_ipp - a3) / a4)
    dt_v_l1 = a1 + a2 * np.cos(x)
    dt_v_l1.loc[abs(x) > np.pi / 2] = a1

    c = 299792458
    dl_v_l1 = c * dt_v_l1
    glo_sats = [sat for sat in sats_aer.index if sat.startswith("R")]
    glo_k = navdata.loc[glo_sats]["FreqNum"]
    glo_sats = glo_k.index.tolist()

    f_l1 = 1575.42
    dl_v = dl_v_l1
    dl_v.loc[glo_sats] = c * dt_v_l1.loc[glo_sats] * f_l1**2 / (1602 + glo_k * 9 / 16)**2

    return dl_v * (1 + 16 * (0.53 - sats_el_sc)**3)

def calc_corr_tropo(station_llh: pd.DataFrame, sats_aer: pd.DataFrame) -> pd.DataFrame:
    """Calculates slant topospheric delays.

    :param station_llh: Observation station coordinates
    :type station_llh: pandas.DataFrame
    :param sats_aer: Horizontal coordinates of satellites
    :type sats_aer: pandas.DataFrame
    :return: Topospheric corrections
    :rtype: pandas.DataFrame
    """
    pgm_path = os.path.join(os.path.dirname(__file__), "egm2008-5.pgm")
    gh = GeoidHeight(pgm_path)
    lat, lon, h = station_llh.to_numpy().flatten()
    undulation = gh.get(float(lat), float(lon))
    h -= undulation
    
    t = 291.16 - 0.0065 * h
    p = 1013.25 * (1 - 2.26e-5 * h)**5.225
    rh = 0.5 * np.exp(-6.3976e-4 * h)
    tc = t - 273.15
    ew = 6.112 * np.exp((17.62 * tc) / (243.12 + tc))
    e = rh * ew

    z = np.radians(90 - sats_aer["elevation"])
    h_km = h / 1000

    b = (
        1.1549
        - 0.1551 * h_km
        + 0.0074 * h_km**2
    )
    dr = (
        -0.0164
        + 0.0027 * h_km
        - 0.00025 * h_km**2
        + (
            0.3773
            - 0.0675 * h_km
            + 0.0043 * h_km**2
        ) / (
            82.7119
            - np.degrees(z)
        )
    )

    return (
        0.002277 / np.cos(z) * (
            p
            + (1255 / t + 0.05) * e
            - b * np.tan(z)**2
        )
        + dr
    )
