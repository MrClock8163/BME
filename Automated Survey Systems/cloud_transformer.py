import argparse
from textwrap import dedent
from copy import deepcopy
from typing import Callable

import numpy as np
import open3d as o3d
import laspy as lp
import pyproj as proj
from pyproj.transformer import TransformerGroup


transformer = Callable[[np.ndarray], np.ndarray]


def load_las(filepath: str) -> lp.LasData:
    with lp.open(filepath) as f:
        las = f.read()
    
    return las


def load_cloud(filepath: str) -> o3d.t.geometry.PointCloud:
    return o3d.t.io.read_point_cloud(filepath)


def write_las(las: lp.LasData, filepath: str):
    las.write(filepath)


def write_cloud(pc: o3d.t.geometry.PointCloud, filepath: str):
    o3d.t.io.write_point_cloud(filepath, pc)


def update_las(las: lp.LasData, points: np.ndarray, crs: proj.CRS) -> lp.LasData:
    # https://www.asprs.org/a/society/committees/standards/asprs_las_format_v12.pdf
    # https://laspy.readthedocs.io/en/latest/intro.html
    x_min, y_min, z_min = np.min(points, axis=0)
    x_max, y_max, z_max = np.max(points, axis=0)
    offset = np.mean(
        np.array(
            (
                (x_min, y_min, z_min),
                (x_max, y_max, z_max)
            )
        ),
        axis=0
    )
    xyz: np.ndarray = np.round((points - offset) * 1000).astype("int32")
    
    las.header.offsets = offset
    las.header.add_crs(crs)
    las.header.x_min = x_min
    las.header.x_max = x_max
    las.header.y_min = y_min
    las.header.y_max = y_max
    las.header.z_min = z_min
    las.header.z_max = z_max
    las.points.X = xyz[:,0]
    las.points.Y = xyz[:,1]
    las.points.Z = xyz[:,2]
    
    return las


def etrs_to_eov(points: np.ndarray) -> np.ndarray:
    etrs = proj.CRS.from_epsg(7931)
    eov = proj.CRS.from_epsg(10660)

    tg = TransformerGroup(etrs, eov)
    if not tg.best_available:
        print("Best transformation is not available, grid files are probably missing.")
        print("Downloading grid files...")
        tg.download_grids(verbose=True)
    
    e, n, z = proj.transform(etrs, eov, points[:,0], points[:,1], points[:,2], always_xy=True)
    return np.column_stack((e, n, z))


def quasi_utm34n_to_eov(points: np.ndarray) -> np.ndarray:
    utm34n = proj.CRS.from_epsg(32634).to_3d()
    wgs84 = proj.CRS.from_epsg(4326).to_3d()
    etrs = proj.CRS.from_epsg(7931)
    eov = proj.CRS.from_epsg(10660)

    tg = TransformerGroup(etrs, eov)
    if not tg.best_available:
        print("Best transformation is not available, grid files are probably missing.")
        print("Downloading grid files...")
        tg.download_grids(verbose=True)
    
    utm2wgs = proj.Transformer.from_crs(utm34n, wgs84, always_xy=True)
    etrs2eov = proj.Transformer.from_crs(etrs, eov, always_xy=True)
    lon, lat, h = utm2wgs.transform(points[:,0], points[:,1], points[:,2])
    e, n, z = etrs2eov.transform(lon, lat, h)

    return np.column_stack((e, n, z))


def transform_las(las: lp.LasData, func: transformer, crs: proj.CRS = None, inplace: bool = False) -> lp.LasData:
    if not inplace:
        las = deepcopy(las)
    
    points_new = func(las.xyz)
    update_las(las, points_new, crs)

    return las


def cli():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=dedent("""
        Transforms a point cloud with UTM34N+H coordinates to Hungarian EOV+EOMA coordinates.
        Transformation is done in two steps:
            1. UTM34N (EPSG:32634) ENH -> WGS84 (EPSG:4326) LLH
            2. WGS84 (EPSG:4326) LLH ~= ETRF2000 (EPSG:7931) LLH -> EOV+EOMA (EPSG:10660) ENH
        """)
    )

    parser.add_argument("input", help="Path to input point cloud.")
    parser.add_argument("output", help="Path to save the transformed point cloud to.")

    args = parser.parse_args()

    cloud = load_las(args.input)
    transform_las(cloud, quasi_utm34n_to_eov, proj.CRS.from_epsg(10660), True)
    write_las(cloud, args.output)


if __name__ == "__main__":
    cli()
