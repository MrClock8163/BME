import argparse
from datetime import timedelta

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import ticker, dates


def load_fixes(filepath: str) -> pd.DataFrame:
    """Loads fix position records from GNSS log file into a dataframe for further processing.

    :param filepath: File path to log file recorded by the GNSSLogger application for Android.
    :type filepath: str or other path like object
    :return: Pandas `DataFrame` object containing the loaded data from fix position records.
        The frame stores data in 5 columns: `"source"` (string), `"time"` (datetime64[s]), `"lat"` (float64), `"lon"` (float64), `"h"` (float64).
    :rtype: pandas.DataFrame
    """
    time: list[float] = []
    source: list[str] = []
    lat: list[float] = []
    lon: list[float] = []
    h: list[float] = []
    with open(filepath) as file:
        for line in file:
            fields = line.strip().split(",")
            if fields[0] != "Fix":
                continue

            source.append(fields[1])
            lat.append(float(fields[2]))
            lon.append(float(fields[3]))
            h.append(float(fields[4]) if fields[4] != "" else np.nan)
            time.append(round(float(fields[8]) / 1000))
    
    df = pd.DataFrame(
        {
            "source": source,
            "time": time,
            "lat": lat,
            "lon": lon,
            "h": h
        }
    )

    return df.astype({"source": "string", "time": "datetime64[s]", "lat": "float64", "lon": "float64", "h": "float64"})


def calc_diff(fixes: pd.DataFrame) -> pd.DataFrame:
    """Calculates the approximate coordinate differences of the GPS and NLP (network) position providers from the FLP (fused) positions.

    The differences are calculated between all FLP positions, and the other providers that logged data in the same epoch.

    :param fixes: GNSS positions logged by all providers (source-time-lat-lon-h).
    :type fixes: pandas.DataFrame
    :return: Pandas `DataFrame` object containing the calculated coordinate differences.
        The frame stores data in 5 columns: `"source"` (string), `"time"` (datetime64[s]), `"dlat"` (float64), `"dlon"` (float64), `"dh"` (float64).
    :rtype: pandas.DataFrame
    """
    sources = []
    dlats = []
    dlons = []
    dhs = []
    times = []

    R = 6_378_000
    meter_per_deg = R * 2 * np.pi / 360
    dlat2m = lambda dlat: dlat * meter_per_deg
    dlon2m = lambda dlon, lat: dlon * meter_per_deg * np.cos(np.radians(lat))

    extras = fixes[fixes["source"] != "FLP"]
    for idx, (source, time, lat, lon, h) in fixes[fixes["source"] == "FLP"].iterrows():        
        concurrent = extras[extras["time"] == time]
        
        if len(concurrent) == 0:
            continue
        
        gps = concurrent[concurrent["source"] == "GPS"]
        nlp = concurrent[concurrent["source"] == "NLP"]

        if len(gps) > 0:
            data = gps.iloc[0]
            sources.append("GPS-FLP")
            dlats.append(dlat2m(data["lat"] - lat))
            dlons.append(dlon2m(data["lon"] - lon, data["lat"]))
            dhs.append(data["h"] - h)
            times.append(time)
        
        if len(nlp) > 0:
            data = nlp.iloc[0]
            sources.append("NLP-FLP")
            dlats.append(dlat2m(data["lat"] - lat))
            dlons.append(dlon2m(data["lon"] - lon, data["lat"]))
            dhs.append(data["h"] - h)
            times.append(time)

    diffs = pd.DataFrame(
        {
            "source": sources,
            "time": times,
            "dlat": dlats,
            "dlon": dlons,
            "dh": dhs
        }
    ).astype(
        {
            "source": "string",
            "time": "datetime64[s]",
            "dlat": "float64",
            "dlon": "float64",
            "dh": "float64"
        }
    )

    return diffs


def calc_availability(fixes: pd.DataFrame, window: int) -> pd.DataFrame:
    """Calculates the fix positions within a sliding window around each epoch during the capture.

    :param fixes: Logged positions (source-time-lat-lon-h).
    :type fixes: pandas.DataFrame
    :param window: Size of the sliding window to use to accumulate positions around each epoch.
        The value should satisfy `window >= 1` and `window % 2 == 1`.
    :type window: int
    :return: Pandas `DataFrame` object containing the accumulated positions for each epoch.
        The frame stores data in 4 columns: `"time"` (datetime64[s]), `"gps"` (uint64), `"nlp"` (uint64), `"flp"` (uint64).
    :rtype: pandas.DataFrame
    """
    dt = timedelta(seconds=(window - 1) // 2)
    start: np.datetime64 = fixes["time"].min()
    end: np.datetime64 = fixes["time"].max()

    time: list[np.datetime64] = []
    gps: list[int] = []
    nlp: list[int] = []
    flp: list[int] = []
    for i in pd.date_range(start, end, freq="s"):
        pos = fixes[((i - dt) <= fixes["time"]) & (fixes["time"] <= (i + dt))]
        time.append(i)
        gps.append(len(pos[pos["source"] == "GPS"]))
        nlp.append(len(pos[pos["source"] == "NLP"]))
        flp.append(len(pos[pos["source"] == "FLP"]))

    df = pd.DataFrame(
        {
            "time": time,
            "gps": gps,
            "nlp": nlp,
            "flp": flp
        }
    ).astype(
        {
            "time": "datetime64[s]",
            "gps": "uint64",
            "nlp": "uint64",
            "flp": "uint64"
        }
    )

    return df


def plot_data(fixes: pd.DataFrame, diffs: pd.DataFrame, av: pd.DataFrame, window: int) -> plt.Figure:
    """Plots logged GNSS position data and differences.

    :param fixes: Logged positions (source-time-lat-lon-h).
    :type fixes: pandas.DataFrame
    :param diffs: Provider differences from fused (source-time-dlat-dlon-dh).
    :type diffs: pandas.DataFrame
    :param av: Availability of positions from providers over time (time-flp-nlp-gps).
    :type av: pandas.DataFrame
    :param window: Size of the sliding window that was used for the availability calculation.
    :type window: int
    :return: Plot figure handle.
    :rtype: matplotlib.pyplot.Figure
    """
    def plot_group(ax: plt.Axes, df: pd.DataFrame, col: str):
        ax.plot(df[df["source"] == "FLP"]["time"], df[df["source"] == "FLP"][col], "-x", color="tab:pink", label="Fused", markersize=2, linewidth=5)
        ax.plot(df[df["source"] == "GPS"]["time"], df[df["source"] == "GPS"][col], "-x", color="lime", label="GPS", markersize=2, linewidth=3)
        ax.plot(df[df["source"] == "NLP"]["time"], df[df["source"] == "NLP"][col], "b-x", label="Network", markersize=2, linewidth=1)

    def plot_diffgroup(ax: plt.Axes, df: pd.DataFrame, col: str):
        ax.plot(df[df["source"] == "GPS-FLP"]["time"], df[df["source"] == "GPS-FLP"][col], "-|", color="lime", label="GPS-Fused", markersize=6, linewidth=3)
        ax.plot(df[df["source"] == "NLP-FLP"]["time"], df[df["source"] == "NLP-FLP"][col], "b-x", label="Network-Fused", markersize=2, linewidth=1)

    fig = plt.figure("GNSS log data", layout="constrained", figsize=(15, 10))
    fig.suptitle("GNSS log data")
    axes: dict[str, plt.Axes] = fig.subplot_mosaic(
        [
            ["lat", "dlat"],
            ["lon", "dlon"],
            ["h", "dh"],
            ["n", "n"]
        ]
    )

    ax11 = axes["lat"]
    ax12 = axes["dlat"]
    ax21 = axes["lon"]
    ax22 = axes["dlon"]
    ax31 = axes["h"]
    ax32 = axes["dh"]
    ax4 = axes["n"]

    ax11.set_title("Logged positions")
    ax11.set_xlabel("Time")
    ax11.set_ylabel("Latitude [°]")
    ax11.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.5f"))
    ax11.xaxis.set_major_formatter(dates.DateFormatter("%H:%M:%S"))
    plot_group(ax11, fixes, "lat")
    ax11.legend()

    ax21.set_xlabel("Time")
    ax21.set_ylabel("Longitude [°]")
    ax21.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.5f"))
    ax21.xaxis.set_major_formatter(dates.DateFormatter("%H:%M:%S"))
    plot_group(ax21, fixes, "lon")
    ax21.legend()

    ax31.set_xlabel("Time")
    ax31.set_ylabel("Altitude [m]")
    ax31.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax31.xaxis.set_major_formatter(dates.DateFormatter("%H:%M:%S"))
    plot_group(ax31, fixes, "h")
    ax31.legend()

    ax12.set_title("Differences of position providers from fused data")
    ax12.set_xlabel("Time")
    ax12.set_ylabel("Latitudinal [m]")
    ax12.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax12.xaxis.set_major_formatter(dates.DateFormatter("%H:%M:%S"))
    plot_diffgroup(ax12, diffs, "dlat")
    ax12.legend()

    ax22.set_xlabel("Time")
    ax22.set_ylabel("Longitudinal [m]")
    ax22.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax22.xaxis.set_major_formatter(dates.DateFormatter("%H:%M:%S"))
    plot_diffgroup(ax22, diffs, "dlon")
    ax22.legend()

    ax32.set_xlabel("Time")
    ax32.set_ylabel("Altitudinal [m]")
    ax32.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax32.xaxis.set_major_formatter(dates.DateFormatter("%H:%M:%S"))
    plot_diffgroup(ax32, diffs, "dh")
    ax32.legend()

    ax4.set_xlabel("Time")
    ax4.set_ylabel("Count")
    if window == 1:
        ax4.set_title("Recorded positions over time")
    else:
        ax4.set_title(f"Recorded positions within {max(window, 1)}s (epoch ± {max((window - 1) // 2, 0)}) sliding window")
    ax4.step(av["time"], av["flp"], "-", color="tab:pink", label="Fused", markersize=2, linewidth=5)
    ax4.step(av["time"], av["gps"], "-", color="lime", label="GPS", markersize=2, linewidth=3)
    ax4.step(av["time"], av["nlp"], "b-", label="Network", markersize=2, linewidth=1)
    ax4.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    ax4.xaxis.set_major_formatter(dates.DateFormatter("%H:%M:%S"))
    ax4.legend()

    return fig


def print_stats(fixes: pd.DataFrame):
    """Prints basic statistics about the logged fix positions in a markdown table format.

    :param fixes: Logged positions (source-time-lat-lon-h).
    :type fixes: pandas.DataFrame
    """
    def print_row(provider: str, data: pd.DataFrame):
        dts = np.diff(data["time"].astype("int64"))
        count = len(data)
        avg = np.average(dts)
        mdn = np.median(dts)

        print(f"| {provider:^8s} | {count:^9d} | {np.min(dts):^11d} | {np.max(dts):^11d} | {avg:^15.0f} | {mdn:^14.0f} |")

    print()
    print("| Provider | Positions | Min gap [s] | Max gap [s] | Average gap [s] | Median gap [s] |")
    print("| :------: | :-------: | :---------: | :---------: | :-------------: | :------------: |")
    print_row("GPS", fixes[fixes["source"] == "GPS"])
    print_row("Network", fixes[fixes["source"] == "NLP"])
    print_row("Fused", fixes[fixes["source"] == "FLP"])


def cli():
    """Command line interface
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("log", metavar="file", help="GNSSLogger file to process")
    parser.add_argument("-o", "--output", help="file to save plot to (the output format is derived from the provided extension)", metavar="FILE", type=str)
    parser.add_argument("-w", "--window", help="sliding window size for position availability visualization", metavar="[1-100]", default=5, type=int, choices=range(1, 31, 2))
    args = parser.parse_args()

    fixes = load_fixes(args.log)
    diff = calc_diff(fixes)
    available = calc_availability(fixes, args.window)
    fig = plot_data(fixes, diff, available, args.window)
    fig.suptitle(args.log)

    print_stats(fixes)

    if args.output:
        fig.savefig(args.output, dpi=300)
        return
    
    plt.show()


if __name__ == "__main__":
    cli()
