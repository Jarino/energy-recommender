import nsgaii.battery as battery


def test_discharge():
    btr = battery.Battery(battery_max=3, current_level=2)

    excess = btr.discharge(5)

    assert btr.current_level == 0
    assert excess == 3


def test_charge():
    btr = battery.Battery(battery_max=3, current_level=0)

    excess = btr.charge(5)

    assert btr.current_level == 3
    assert excess == 2
