from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def skyplot(
    sats: pd.DataFrame,
    station: pd.DataFrame,
    epoch: datetime,
    cutoff: float,
    filter_cutoff: bool = False
):
    """Generates satellite skyplot from topocentric horizontal coordinates.

    :param sats: `azimut-elevation-range` data
    :type sats: pandas.DataFrame
    :param station: ˙x-y-z` or `lat-lon-h` data of the observation station
    :type station: pandas.DataFrame
    :param cutoff: elevation cutoff angle
    :type cutoff: float
    :param filter_cutoff: hide satellites under the cutoff angle
    :type filter_cutoff: bool
    """
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

    visible = sats[sats["elevation"] > (0 if not filter_cutoff else cutoff)]

    satellites = visible.index.get_level_values("sv").tolist()
    sats_gps = list(filter(lambda n: n[0] == "G", satellites))
    sats_gal = list(filter(lambda n: n[0] == "E", satellites))
    sats_bd = list(filter(lambda n: n[0] == "C", satellites))
    sats_glo = list(filter(lambda n: n[0] == "R", satellites))

    a = np.radians(visible["azimut"])
    e = 90 - visible["elevation"]

    for name, sv, marker in zip(
        ["GPS", "Galileo", "BeiDou", "GLONASS"],
        [sats_gps, sats_gal, sats_bd, sats_glo],
        ["og", "ob", "oy", "or"]
    ):
        ax.plot(a.loc[sv], e.loc[sv], marker, markersize=8, label=name)

    for sat_a, sat_e, sat_id in zip(a, e, visible.index):
        ax.annotate(
            sat_id,
            xy=(sat_a, sat_e),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center"
        )
    ax.margins(0)
    ax.fill_between(
        np.linspace(0, 2*np.pi, 100),
        90,
        90 - cutoff,
        color='red',
        alpha=0.2,
        label=f"elevation cutoff: {cutoff:.0f}°"
    )

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rticks((90, 60, 30, 0))
    ax.set_rgrids((0, 30, 60, 90), labels=("90°", "60°", "30°", "0°"))
    ax.set_rlabel_position(45)
    ax.set_thetagrids(
        np.rad2deg(
            np.linspace(0, 2 * np.pi, 5)
        )[1:],
        labels=("E", "S", "W", "N")
    )

    s1, s2, s3 = station.iloc[0, :].to_numpy()

    if "x" in station.columns:
        ax.set_title(
            f"Skyplot at {station.index[0]} at {epoch} (UTC)\n"
            f"(X: {s1:.3f}, Y: {s2:.3f}, "
            f"Z: {s3:.3f})"
        )
    elif "lat" in station.columns:
        ax.set_title(
            f"Skyplot at {station.index[0]} at {epoch} (UTC)\n"
            f"(Lat.: {s1:.5f}, Lon.: {s2:.5f}, "
            f"H: {s3:.3f})"
        )

    ax.legend(loc="lower right")
    plt.show()
