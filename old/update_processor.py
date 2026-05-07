"""
Update Processor: Detects new sensor measurements and calls filter updates.

Simple processor that iterates through flight data, detects when measurements
change, and triggers the appropriate update function.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


class UpdateProcessor:
    """
    Detects sensor measurement changes and triggers filter updates.
    """

    def __init__(self, csv_path: str):
        """
        Initialize with flight data.

        Args:
            csv_path: Path to flight data CSV file
        """
        self.df = pd.read_csv(csv_path)
        self.current_row = 0
        self.last_gnss = None
        self.last_mag = None
        self.last_baro = None
        self.gnss_ref_lat = None  # Reference latitude for NED conversion
        self.gnss_ref_lon = None  # Reference longitude for NED conversion
        self.gnss_ref_alt = None  # Reference altitude for NED conversion

    def get_next_row(self) -> Tuple[int, pd.Series]:
        """
        Get next row from data.

        Returns:
            (row_index, row_data)

        Raises:
            StopIteration when end of data
        """
        if self.current_row >= len(self.df):
            raise StopIteration

        row = self.df.iloc[self.current_row]
        row_idx = self.current_row
        self.current_row += 1
        return row_idx, row

    def check_gnss_update(self, row: pd.Series) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Check if GNSS measurement is new (using GPS columns).

        Returns:
            (relative_position_ned, velocity_ned) if new, None otherwise
        """
        gps_lat = row['gps_lat(deg)']
        gps_lon = row['gps_long(deg)']
        gps_alt = row['gps_alt(mm)'] / 1000.0  # Convert mm to m
        gps_vn = row['gps_vn(mm/s)'] / 1000.0  # Convert mm/s to m/s
        gps_ve = row['gps_ve(mm/s)'] / 1000.0
        gps_vd = row['gps_vd(mm/s)'] / 1000.0

        # Initialize reference on first valid GPS reading
        if self.gnss_ref_lat is None:
            self.gnss_ref_lat = gps_lat
            self.gnss_ref_lon = gps_lon
            self.gnss_ref_alt = gps_alt
            self.last_gnss = (gps_lat, gps_lon, gps_alt)
            # Convert to relative NED
            rel_pos = self._lat_lon_alt_to_ned(gps_lat, gps_lon, gps_alt)
            vel = np.array([gps_vn, gps_ve, gps_vd])
            return rel_pos, vel

        # Check if GPS position has changed from last update
        last_lat, last_lon, last_alt = self.last_gnss
        if not (np.isclose(gps_lat, last_lat, atol=1e-7) and
                np.isclose(gps_lon, last_lon, atol=1e-7) and
                np.isclose(gps_alt, last_alt, atol=0.1)):
            self.last_gnss = (gps_lat, gps_lon, gps_alt)
            # Convert to relative NED
            rel_pos = self._lat_lon_alt_to_ned(gps_lat, gps_lon, gps_alt)
            vel = np.array([gps_vn, gps_ve, gps_vd])
            return rel_pos, vel

        return None

    def _lat_lon_alt_to_ned(self, lat: float, lon: float, alt: float) -> np.ndarray:
        """
        Convert lat/lon/alt to relative NED coordinates from reference point.

        Args:
            lat: Latitude (degrees)
            lon: Longitude (degrees)
            alt: Altitude (meters)

        Returns:
            Relative position in NED frame (meters)
        """
        # Earth radius
        R_earth = 6371000.0

        # Convert to radians
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        ref_lat_rad = np.radians(self.gnss_ref_lat)
        ref_lon_rad = np.radians(self.gnss_ref_lon)

        # North: change in latitude
        d_lat = (lat_rad - ref_lat_rad) * R_earth

        # East: change in longitude (scaled by cos of latitude)
        d_lon = (lon_rad - ref_lon_rad) * R_earth * np.cos(ref_lat_rad)

        # Down: negative change in altitude
        d_alt = self.gnss_ref_alt - alt

        return np.array([d_lat, d_lon, d_alt])

    def check_magnetometer_update(self, row: pd.Series) -> Optional[np.ndarray]:
        """
        Check if magnetometer measurement is new.

        Returns:
            mag_field if new, None otherwise
        """
        mag = np.array([row['mx(G)'], row['my(G)'], row['mz(G)']])

        if self.last_mag is None:
            self.last_mag = mag.copy()
            return mag

        if not np.allclose(mag, self.last_mag, atol=0.001):
            self.last_mag = mag.copy()
            return mag

        return None

    def check_barometer_update(self, row: pd.Series) -> Optional[float]:
        """
        Check if barometer measurement is new.

        Returns:
            altitude if new, None otherwise
        """
        alt = row['baro_alt(m)']

        if self.last_baro is None:
            self.last_baro = alt
            return alt

        if not np.isclose(alt, self.last_baro, atol=0.01):
            self.last_baro = alt
            return alt

        return None

    def has_data(self) -> bool:
        """Check if more data to process."""
        return self.current_row < len(self.df)
