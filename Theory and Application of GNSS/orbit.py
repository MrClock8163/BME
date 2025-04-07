import os
from datetime import datetime

import pandas as pd

from gnsscalc import load_coords
from gnsscalc.transform import Ellipsoid, Transformer
from gnsscalc.satnav import Orbit


def main():
    epoch = datetime(2025, 2, 10, 7, 15, 0)
    savedir = "data"

    orb = Orbit(epoch, savedir)
    orb.load_usable_navdata()
    orb.calc_sat_ecef_xyz()

    station = ("LICC00GBR", 3978703.5690, -12141.0380, 4968417.2180)
    station_df = load_coords(
        [
            station
        ],
        'XYZ'
    )

    ecef = orb.sat_xyz
    transf = Transformer(Ellipsoid.wgs84())
    aer = transf.xyz2aer(ecef, station_df)

    coords = pd.concat([ecef, aer], axis=1)
    coords.to_csv(
        os.path.join(
            savedir,
            f"satcoords_{station[0]:s}_{epoch:%Y%m%d_%H%M%S}.csv"
        )
    )

    corr = orb.calc_atmospheric_delays(transf.xyz2llh(station_df), coords)
    complete_data = pd.concat([coords, corr], axis=1)
    visible_corr = complete_data[complete_data["elevation"] >= 10][["elevation", "delay_iono", "delay_tropo"]]

    visible_corr.to_csv(
        os.path.join(
            savedir,
            f"satcorrs_{station[0]:s}_{epoch:%Y%m%d_%H%M%S}.csv"
        )
    )


if __name__ == "__main__":
    main()
