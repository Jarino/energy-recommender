"""
Modul for calculation of cost of energy for day,
based on price table
"""

import numpy as np

BASE = 0.05
WINTER_ON_PEAK = 0.03316
WINTER_OFF_PEAK = -0.01125

SUMMER_ON_PEAK = 0.06124
SUMMER_OFF_PEAK = WINTER_OFF_PEAK


def calculate(x, winter=False, weekend=False):
    """
    calculates price of energy for given vector, based on wheter
    its weekend, weekday, summer or winter
    """
    x = np.array(x)
    prices = []

    if weekend:
        # weekend prices are same for winter and summer
        prices = np.repeat(BASE + SUMMER_OFF_PEAK, 24)
    elif winter:
        prices = np.repeat(BASE + WINTER_OFF_PEAK, 24)
        prices[6:10] = BASE + WINTER_ON_PEAK
        prices[17:20] = BASE + WINTER_ON_PEAK
    else:  # summer
        prices = np.repeat(BASE + SUMMER_OFF_PEAK, 24)
        prices[16:20] = BASE + SUMMER_ON_PEAK

    return sum(x * prices)
