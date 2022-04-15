import jumpy as jp

from flax import struct


@struct.dataclass
class ForceConfig(object):
    """
    Velocities of some object with respect to some frame.
    """
    strength: jp.ndarray
    body: str
