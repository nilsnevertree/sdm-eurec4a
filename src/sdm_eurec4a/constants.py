"""
Module containing constants used in the project.
"""

from typing import Any


class WaterConstants:
    # density of water
    density = 1000  # kg / m^3
    # latent heat of vaporization of water
    vapourization_heat = 2257e3  # J / kg


class TimeSlices:
    quasi_stationary_state = slice(1500, 3598)
    full_state = slice(0, 3598)
