import os
from datetime import datetime

import pandas as pd

from gnsscalc import load_coords
from gnsscalc.transform import Ellipsoid, Transformer
from gnsscalc.satnav import Orbit
from gnsscalc.visualize import skyplot


def main():
    epoch = datetime(2025, 2, 10, 7, 15, 0)
    savedir = "data"

    orb = Orbit(epoch, savedir)
    orb.load_usable_navdata()
    orb.calc_sat_ecef_xyz()

    london = load_coords(
        [
            ("LICC00GBR", 3978703.5690, -12141.0380, 4968417.2180)
        ],
        'XYZ'
    )

    ecef = orb.sat_xyz
    transf = Transformer(Ellipsoid.wgs84())
    aer = transf.xyz2aer(ecef, london)

    coords = pd.concat([ecef, aer], axis=1)
    coords.to_csv(
        os.path.join(
            savedir,
            f"satcoords_LICC00GBR_{epoch:%Y%m%d_%H%M%S}.csv"
        )
    )

    skyplot(coords, london, epoch, 10, True)


if __name__ == "__main__":
    main()
