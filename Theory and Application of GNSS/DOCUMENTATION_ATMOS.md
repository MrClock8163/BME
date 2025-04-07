# Theory and Application of GNSS - Homework Assignment 4

Zalán Janurik

2024/25/2

## 1. Station and epoch

Details of chosen IGS station:

|                  |                                   |
| ---------------- | --------------------------------- |
| Website link     | https://network.igs.org/LICC00GBR |
| ID               | LICC00GBR                         |
| Location         | London, Imperial College          |
| Log last updated | 2024.12.16.                       |

ITRF station coordinates:

|        X [m] |       Y [m] |        Z [m] |
| -----------: | ----------: | -----------: |
| 3978703.5690 | -12141.0380 | 4968417.2180 |

The chosen epoch is the start of the first class of the 2024/25/2
semester in UTC time:

- **2025.02.10. 07:15:00**

## 2. Satellite coordinates

The horizontal coordinates of the satellites at the observation
epoch are calculated with the methods implemented in the previous
assignments.

## 3. Ionospheric delays

The ranging effect of the ionospheric delay was calculated
with the Klobuchar model. The process was done in the following
general steps:

1. Geographic coordinates of the ionospheric piercing point (IPP)
2. Geomagnetic coordinates of the IPP
3. Vertical delay at the IPP for GPS L1 frequency
4. Vertical delay at the IPP for individual Glonass satellites
5. Slant delay from vertical delay, using a mapping function

The results are given as the effect of the time delay on the
ranging in meters.

## 4. Tropospheric delays

The effect of the tropospheric delay was calculated with the
modified Saastamoinen model.

To convert the ellipsoidal height of the receiver to orthometric
height. The geoid undulation was computed using a
[custom class](https://github.com/vandry/geoidheight/blob/f7df936bf5856bff935802be523b39dda6b89620/geoid.py) (written by
Kim Vandry) with the EGM2008 model.

General calculation steps:

1. Meteorological parameters from standard atmosphere model
2. Correction terms related to orthometric height and zenith angle
3. Slant delay from computed parameters

The Saastamoinen model gives the slant delay effect, without the
need to transform it using mapping functions.

## 5. Export

The calculated atmospheric delay effects can be joined into a single
data block, and exported to a `.csv` file.

(Only satellites above 10° elevation were exported.)
