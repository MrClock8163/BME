# Theory and Application of GNSS - Homework Assignment 3

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

## 2. Downloading the navigation data

Once a `datetime` object with the observation epoch is created,
the `Orbit` class can be instantiated. During the initialization,
the `Orbit` automatically downloads and decompresses the appropriate
RINEX navigation file from the FTP server of the
[BKG GNSS Data Center](https://igs.bkg.bund.de/root_ftp/IGS/BRDC/) to
the specified save directory.

```py
epoch = datetime(2025, 2, 10, 7, 15, 0)
savedir = "data"
orb = Orbit(epoch, savedir)
```

## 3. Loading the navigation data

From the downloaded RINEX file, the satellite navigation parameters
can be read, but only those, that are indicated as "healthy" and are
close enough in time to the chosen epoch.

Since the GPS, Galileo and BeiDou systems use similar parameter set,
with slightly different constants, the reading can be done in two main
parts.

### 3.1 Loading GPS, Galileo and BeiDou navigation data

From the GPS, Galileo and BeiDou records the ones with an other than `0`
health parameter (`health`, `health` and `SatH1` respectively) are
dropped. Furthermore, only those parameter sets are kept that are in the
+/- 4 hours time interval around the observation epoch. From the
remaining data, the parameters closest to the observation epoch are
chosen for every usable satellite.

```py
gps_data = self.read_usable_navdata('G')
gal_data = self.read_usable_navdata('E')
bd_data = self.read_usable_navdata('C')
```

### 3.2 Loading GLONASS navigation data

Since the GLONASS satellites use a different parameter set, the loading
is filtering is also slightly different. Like for the other constellations
the unhealthy data sets are dropped, then the sets outside a +/- 15 minutes
interval around the epoch are also dropped. From the remaining data sets
the one closest to the epoch is chosen for every satellite.

```py
glo_data = self.read_usable_navdata_glonass()
```

## 4. ECEF satellite coordinates

Like the data loading and filtering, the calculation of the ECEF XYZ
coordinates of the satellites is also done differently for the
GPS-Galileo-BeiDou and the GLONASS constellations.

The calculated coordinates are stored in the `Orbit` object for later access.

```py
def calc_sat_ecef_xyz(self):
    ecef = self.calc_ecef_xyz_gps_gal_bd()
    ecef_glo = self.calc_ecef_xyz_glo()
    self.sat_xyz = pd.concat([ecef, ecef_glo], axis=0)
```

### 4.1 Orbit calculations for GPS, Galileo and BeiDou

The calculations for these three systems are indentical, they only differ
in the applied constants. The calculations consist of the following
general steps:

1. initial correction time from observation and data set reference times
2. initial satellite clock corrections without relativistic effect
3. modified correction time
4. mean anomaly and eccentric from parameters through Kepler equations
5. relativistic correction from eccentric anomaly
6. new corrections with relativistic correction
7. orbital coordinates
8. ECEF XYZ coordinates from orbital coordinates

### 4.2 Orbit calculation for GLONASS

The GLONASS system uses an entirely different method of calculation. From
the broadcast parameters, the satellite coordinates can be calculated by
numerical integration, using the 4th order Runge-Kutta method. In contrast
to the other constellatiosn, due to the calculatiosn can only be done with
parameters not older/newer than 15 minutes.

The coordinates are calculated with an iterative numerical method using
the broadcast satellite coordinates, velocities and accelerations. In each
step a set of ordinary differential equations needs to be solved, that yields
the new estimated values. With a step size of 10 seconds, the number of steps
depends on the correction time.

This method does not require an additional transformation to ECEF coordinates,
as the methods already yields them as such.

## 5. Visualization

In order to visalize the satellites on a skyplot at the chosen IGS station,
the ECEF coordinates need to be transformed to a horizontal coordinate
system.

To transform the coordinates, a helper class can be created with the
appripriate ellipsoid definition.

```py
transf = Transformer(Ellipsoid.wgs84())
```

The `Transformer` object can then be used to transform the ECEF coordinates
to horizontal coordinates around the chosen station. The function returns
both the topocentric, and the horizontal coordinates.

```py
aer = transf.xyz2aer(ecef, london)
```

Once the horizontal coordinates are calculated, a satellite skyplot can be
generated.

```py
skyplot(coords, london, epoch, 10, True)
```

## 6. Export

Finally the ECEF, topocentric and horizontal coordinates can be joined
into a single data block, and exported as a simple `*.csv` file.
