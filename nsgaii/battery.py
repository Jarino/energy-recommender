class Battery():

    def __init__(self, battery_max=3, current_level=0):
        self.battery_max = battery_max
        self.current_level = current_level

    def charge(self, charge):
        """charge battery with specified energy
        returns excess amount of energy
        """
        if charge + self.current_level > self.battery_max:
            self.current_level = self.battery_max
        else:
            self.current_level += charge

        return charge - self.current_level

    def discharge(self, discharge):

        if discharge >= self.current_level:
            diff = discharge - self.current_level
            self.current_level = 0
            return diff

        self.current_level -= discharge

        return 0
