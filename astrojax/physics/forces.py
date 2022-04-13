from abc import ABC, abstractmethod
from typing import List, Tuple

from astrojax import pytree
import jax.numpy as jp
import jax

from astrojax.state import TimeDerivatives
from astrojax.state import PosVel
from astrojax.physics.bodies import Body
from flax import struct


@struct.dataclass
class Force(object):
    """
    Velocities of some object with respect to some frame.
    """
    strength: jp.ndarray
    body: str
