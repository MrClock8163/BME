# Theory and Application of GNSS - Homework Assignment 5

Zalán Janurik

2024/25/2

## 1. Station and epoch

Details of chosen EUREF station:

|                  |                                                                         |
| ---------------- | ----------------------------------------------------------------------- |
| Website link     | https://epncb.eu/_networkdata/siteinfo4onestation.php?station=BUTE00HUN |
| ID               | BUTE00HUN                                                               |
| Location         | Budapest, Budapest University of Technology and Economics               |
| Log last updated | 2024.07.23.                                                             |

ITRF station coordinates:

|       X [m] |       Y [m] |       Z [m] |
| ----------: | ----------: | ----------: |
| 4081882.424 | 1410011.131 | 4678199.424 |

The chosen epoch is the start of the first class of the 2024/25/2
semester in GPS time:

- **2025.02.10. 07:15:00**

## 2. Downloading the navigation data

The navigation broadcast datasets can be retrieved from the
archives of the different data centers like BKG GNSS Data Center.
The data is stored in RINEX files.

The files are Gzip compressed, which can be decompressed with the
`gzip` package, that is part of the Python standard library. The
decompressed RINEX files then can be parsed with the `georinex`
package.

## 3. Downloading the observation data

Similarly to the navigation files, the observation files can also
be downloaded from GNSS archives. The BKG archive serves observation
data from a number of different observation station networks, like
IGS, EUREF and more. The observations are available in compressed
RINEX format, further compressed with Gzip.

The Gzip compression can be reversed with the `gzip` module of the
standard library (similar to how it can be done to the broadcast
datasets). The extracted file is still compressed with Hatanaka
compression, that has to be reversed, to get the normal RINEX file.
The conversion can be done with the `hatanaka` python package, that
is also a standard dependency of the `georinex` package, so it does
not create additional dependencies.

The prepared RINEX file can be read with the `georinex` package.

## 4. Preliminary calculations

Before the receiver position can be calculated from the code
measurements, a number of parameters must be prepared.

### 4.1 Satellite coordinates

The satellite coordinates have to be calculated for the time of
the transmission, not the reception. The approximate time of the
transmission can be derived from the pseudoranges, by calculating
the propagation time, and subtracting it from the reception epoch.

The calculated ECEF coordinates then have to be rotated in order
to account for the rotation of the Earth during the signal
propagation time.

### 4.2 Error corrections

Certain error correction terms must be calculated parallel to the
satellite coordinate calculcation. These are the satellite clock
errors and the relativistic errors.

The ionospheric and tropospheric corrections can be calculated
from the parameters included in the navigation datasets.

## 5. Position calculation

The receiver position can be calculated with a least squares
adjustment, using the linearized pseudorange equations. Inputs are:

- setallite ECEF coordinates (at time of transmission)
- error correction terms (atmoshperic, clock, etc.)
- preliminary receiver position

In total, 7 parameters must be adjusted:

- 3 receiver coordinates
- 4 receiver clock errors (1 for each satellite constellation)

From these inputs the design matrix can be constructed, to serve
as the basis of the adjustment calculation.

Since the elevation angle of the satellites has a great effect
on the amtospheric errors, the measurements are weighted by the
elevations of the satellites.

Since the pseudorange equations are not linear in their true form,
the linearized forms must be used in the adjustment. This also
means, that the solution must be done iteratively.

## 6. Results

The adjustment yields the changes of the parameter values, from
which the adjusted parameters can be calcuated. From the covariace
matrix the mean errors of the parameters can be also calculated.
