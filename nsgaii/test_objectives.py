import numpy as np

import nsgaii.objectives as objectives
from nsgaii.battery import Battery


def test_consumption_with_battery():
    energy_excess = np.array([0, 0, 5, 5, 0, 0, 0])
    consumption = np.array([2, 1, 0, 0, 5, 3, 2])
    expected_from_grid = np.array([2, 1, 0, 0, 2, 3, 2])

    btr = Battery(3, 0)

    from_grid = objectives.consumption_with_battery(consumption,
                                                    energy_excess, btr)

    np.testing.assert_array_equal(expected_from_grid, from_grid)


def test_consumption_with_battery_with_remains():
    energy_excess = np.array([0, 0, 5, 5, 0, 0, 0])
    consumption = np.array([2, 1, 0, 0, 1, 3, 2])
    expected_from_grid = np.array([2, 1, 0, 0, 0, 1, 2])

    btr = Battery(3, 0)

    from_grid = objectives.consumption_with_battery(consumption,
                                                    energy_excess, btr)

    np.testing.assert_array_equal(expected_from_grid, from_grid)
