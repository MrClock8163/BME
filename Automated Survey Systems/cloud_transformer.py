import argparse
from textwrap import dedent

import numpy as np
import open3d as o3d
import laspy as lp
import pyproj as proj
from pyproj.transformer import TransformerGroup


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
    offset = np.mean(points, axis=0)
    xyz: np.ndarray = np.round((points - offset) * 1000).astype("int32")
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    
    las.header.offsets = offset
    las.header.add_crs(crs)
    las.header.x_min = np.min(x)
    las.header.x_max = np.max(x)
    las.header.y_min = np.min(y)
    las.header.y_max = np.max(y)
    las.header.z_min = np.min(z)
    las.header.z_max = np.max(z)
    las.points.X = x
    las.points.Y = y
    las.points.Z = z
    
    return las


def utm34n_to_eov(points: np.ndarray) -> np.ndarray:
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
    points_eov = utm34n_to_eov(cloud.xyz)
    update_las(cloud, points_eov, proj.CRS.from_epsg(10660))
    write_las(cloud, args.output)


if __name__ == "__main__":
    cli()

# https://www.asprs.org/a/society/committees/standards/asprs_las_format_v12.pdf
# https://laspy.readthedocs.io/en/latest/intro.html
