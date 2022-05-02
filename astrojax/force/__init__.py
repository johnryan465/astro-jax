from abc import ABC


class ForceModel(ABC):
    """
    A force model given certain objects creates a force function which can be used to calculate the force between the objects.
    """
