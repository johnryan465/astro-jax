from astrojax.force import ForceModel
from astrojax.objects.celestial import Celestial
from astrojax.objects.spacecraft import Spacecraft


class GravityForceModel(ForceModel):
    """
    Calculates the force between two objects.
    """

    def __init__(self, celestial: Celestial, craft: Spacecraft):
        self.celestial = celestial
        self.craft = craft
