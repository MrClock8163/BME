from datetime import datetime, timedelta
from typing import Literal
from textwrap import dedent

import numpy as np
import pandas as pd
import scipy.stats
import georinex as gr

from gnsscalc import (
    download_obs,
    extract_gzip,
    decompress_crx,
    download_brdc,
    load_coords
)
import gnsscalc.satnav as sat
import gnsscalc.transform as trans


SOL = 299792458 # speed of light
OMEGA = 7.2921151467E-5 # Earth rotation [rad/s]


def download_datasets(
    epoch: datetime,
    network: Literal['EUREF', 'IGS'],
    station: str,
    savedir: str
) -> tuple[str, str]:
    """Downloads and extracts navigation and observation RINEX files (skips if they already exist)"""

    path_nav_gz = download_brdc(epoch, savedir)
    path_nav = extract_gzip(path_nav_gz)

    path_obs_comp_gz = download_obs(network, station, epoch, savedir)
    path_obs_comp = extract_gzip(path_obs_comp_gz)
    path_obs = decompress_crx(path_obs_comp)

    return path_nav, path_obs


def read_pseudoranges(epoch: datetime, path_obs: str) -> pd.Series:
    """Loads pseudorange values from the provided observation RINEX"""
    t_start = epoch - timedelta(seconds=30)
    t_end = epoch + timedelta(seconds=30)

    obs_data = gr.load(
        path_obs,
        use=("G", "E", "R", "C"),
        meas=("C1C", "C1P"),
        tlim=(t_start, t_end)
    )
    obs_data_df = obs_data.to_dataframe().dropna(how="all")
    meas = obs_data_df.xs(epoch)
    return meas.C1C.fillna(0) + meas.C1P.fillna(0)


def apply_earth_rotation(row: pd.Series) -> pd.Series:
    """Applies the effect of the Earth rotation to satellite coordinates"""
    tau = row[0]
    rz = np.array([
        [np.cos(tau * OMEGA), np.sin(tau * OMEGA), 0], 
        [-np.sin(tau * OMEGA), np.cos(tau * OMEGA), 0],
        [0, 0, 1]
    ])
    row.loc[["x", "y", "z"]] = np.dot(rz, row.loc[["x", "y", "z"]])
    return row


def adjust_receiver_coordinates(
    preliminary_receiver_pos: np.ndarray,
    pseudoranges: pd.Series,
    sat_coords: pd.DataFrame,
    sat_aux: pd.DataFrame,
    atmospherics: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs least squares adjustment on receiver coordinates (adapted from lecture handout notebook)"""

    # Number of parameters: 3 coordinates + 4 receiver clock offsets = 7
    num_param = 7

    # Arrays to hold the parameter values and the corrections.
    param_val = np.zeros([num_param, 1])
    param_corr = np.full([num_param, 1], np.inf)

    # The initial position of the receiver (from the observation file header).
    rec_pos = preliminary_receiver_pos[:, np.newaxis].copy() # converted to column vector

    # Create a copy of the satellite coordinates DataFrame that we can modify by dropping 
    # outliers in the calculation.
    sat_coords_calc = sat_coords.copy(deep=True)

    # Iterate until the parameters corrections become less than the tolerance.
    iteration = 0
    while abs(np.linalg.norm(param_corr)) > 10e-2:
        # We compute the number of satellites inside the loop as it can change due to the error 
        # screening at the end.
        num_sat = sat_coords_calc.shape[0]

        # Initializing the design matrix, weight vector and the right hand side vector.
        design_mat = np.zeros([num_sat, num_param])
        obs_vec = np.zeros([num_sat, 1])
        weights = np.diag(np.zeros([num_sat]))

        # Looping through each satellite and filling up the values.
        for i in range(num_sat):
            # Constellation id from the satellite id.
            sat_id = sat_coords_calc.index[i]

            # Set up the rec. clock offset columns.
            match sat_id[0]:
                case 'G':
                    receiver_clock_column = 3
                    receiver_clock_offset = param_val[receiver_clock_column]
                case 'E':
                    receiver_clock_column = 4
                    receiver_clock_offset = param_val[receiver_clock_column]
                case 'C':
                    receiver_clock_column = 5
                    receiver_clock_offset = param_val[receiver_clock_column]
                case 'R':
                    receiver_clock_column = 6
                    receiver_clock_offset = param_val[receiver_clock_column]

            # Corrected pseudorange.
            corrected_pseudorange = (
                pseudoranges[sat_id]
                + SOL * sat_aux.delta_s[sat_id]
                + SOL * sat_aux.delta_rel[sat_id]
                + SOL * sat_aux.tgd[sat_id]
                - receiver_clock_offset
                - atmospherics.iono[sat_id]
                - atmospherics.tropo[sat_id]
            )

            # Receiver-satellite unit vectors.
            dv = (
                sat_coords_calc.loc[sat_id].loc[["x", "y", "z"]]
                - rec_pos.T.flatten()
            ) # difference vector
            geometric_distance = np.linalg.norm(dv) # geometric distance
            uv = -dv / geometric_distance # unit vector

            # Design matrix.
            design_mat[i, 0:3] = uv
            design_mat[i, receiver_clock_column] = 1

            # Right hand side vector.
            obs_vec[i] = corrected_pseudorange - geometric_distance

            # Weight matrix
            mu2 = 1 / (np.sin(np.radians(sat_coords.elevation[sat_id]))**2)
            weights[i, i] = 1 / mu2

        # Solution with least squares.
        param_corr = (
            np.linalg.pinv(design_mat.T @ weights @ design_mat)
            @ (design_mat.T @ weights @ obs_vec)
        )

        # Error screening.
        r = obs_vec - design_mat @ param_corr
        mad = scipy.stats.median_abs_deviation(r)
        isoutlier = (abs(r - np.median(r)) > 4*mad).flatten()
        if any(isoutlier):
            # Print the outliers.
            print(
                f"Outliers in iteration {iteration + 1:d}: "
                f"{sat_coords_calc.index[isoutlier].to_numpy()}"
            )

            # Remove the satellites from the computation.
            not_outlier_id = sat_coords_calc.index[np.invert(isoutlier).flatten()]
            sat_coords_calc = sat_coords_calc.loc[not_outlier_id]

            # Reset the parameter corrections.
            param_corr = np.full([num_param, 1], np.inf)

            # Restart the loop.
            iteration += 1
            continue

        # If there are no outliers, update the parameter values.
        param_val += param_corr
        rec_pos += param_corr[0:3]

        # Increase the number of iterations.
        iteration += 1

    print(f"Finished in {iteration:d} iterations")

    # The covariance matrix of the parameters.
    sigma2 = r.T @ weights @ r / (sat_coords_calc.size - num_param)
    cov = sigma2 * np.linalg.pinv(design_mat.T @ weights @ design_mat)

    return rec_pos, param_val, cov


def display_results(
    station_name: str,
    prelim_rec_pos: np.ndarray,
    rec_pos: np.ndarray,
    params: np.ndarray,
    cov: np.ndarray
):
    """Formats and prints adjustments results to STDOUT"""

    params = params.flatten()
    rec_pos = rec_pos.flatten()
    print(
        dedent(
            f"""
            Station: {station_name}

            Preliminary receiver coordinates:
            - x = {prelim_rec_pos[0]:.3f}
            - y = {prelim_rec_pos[1]:.3f}
            - z = {prelim_rec_pos[2]:.3f}

            Final parameters:
            - dx = {params[0]:.3f}
            - dy = {params[1]:.3f}
            - dz = {params[2]:.3f}
            - c * clock_GPS = {params[3]:f}
            - c * clock_GAL = {params[4]:f}
            - c * clock_BEI = {params[5]:f}
            - c * clock_GLO = {params[6]:f}

            Final receiver coordinates:
            - x = {rec_pos[0]:.3f}
            - y = {rec_pos[1]:.3f}
            - z = {rec_pos[2]:.3f}

            """
        )
    )
    print("Covariance matrix:")
    with np.printoptions(precision=3):
        print(cov)

    rms = np.sqrt(np.diag(cov)[:3]).flatten()
    print(
        dedent(
            f"""
            
            Position accuracy:
            - sigma X = {rms[0]:.3f}
            - sigma Y = {rms[1]:.3f}
            - sigma Z = {rms[2]:.3f}

            3D position accuracy: {np.sqrt(np.diag(cov)[:3].sum()):.3f}
            """
        )
    )


def main():
    epoch_gpst = datetime(2025, 2, 10, 7, 15, 0)
    epoch_utc = epoch_gpst - timedelta(seconds=sat.GPST_LEAP)
    transformer = trans.Transformer(trans.Ellipsoid.wgs84())

    # Retrieving RINEX datasets
    station_name = "BUTE00HUN"
    network = "EUREF"
    path_nav, path_obs = download_datasets(
        epoch_utc,
        network,
        station_name,
        "data"
    )

    # Loading preliminary position
    header = gr.rinexheader(path_obs)
    pos0 = np.array(header["position"])
    station = load_coords(
        [
            (station_name, *pos0)
        ],
        "XYZ"
    )
    station_llh = transformer.xyz2llh(station)

    pseudoranges = read_pseudoranges(epoch_gpst, path_obs)

    # Reading and preparing satellite navigation data
    propagation_times = pseudoranges / SOL
    transmission_epochs = epoch_utc - pd.to_timedelta(
        propagation_times,
        "s"
    )
    navdata = sat.load_usable_navdata(path_nav, epoch_utc)
    ecef_preliminary, sat_corrections = sat.calc_sat_ecef(
        transmission_epochs,
        navdata
    )
    ecef_preliminary_tau = pd.concat(
        [
            ecef_preliminary,
            propagation_times
        ],
        axis=1
    )
    aer_preliminary = transformer.xyz2aer(ecef_preliminary, station)
    aer_preliminary_cutoff = aer_preliminary[aer_preliminary["elevation"] >= 15]
    sat_coords = pd.concat(
        [
            ecef_preliminary_tau.apply(apply_earth_rotation, axis=1).drop(0, axis=1),
            aer_preliminary_cutoff["elevation"]
        ],
        axis=1
    ).dropna(how="any")

    # Calculating atmospheric correction terms
    atmos_corr = pd.concat(
        [
            sat.calc_corr_iono(
                path_nav,
                epoch_gpst,
                navdata,
                station_llh,
                aer_preliminary_cutoff
            ),
            sat.calc_corr_tropo(station_llh, aer_preliminary_cutoff)
        ],
        axis=1
    )
    atmos_corr.columns = ["iono", "tropo"]

    # Running least squares adjustment
    receiver, parameters, covariance = adjust_receiver_coordinates(
        pos0,
        pseudoranges,
        sat_coords,
        sat_corrections,
        atmos_corr
    )
    display_results(station_name, pos0, receiver, parameters, covariance)


if __name__ == "__main__":
    main()
