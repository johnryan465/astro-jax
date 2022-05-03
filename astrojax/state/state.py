from typing import Tuple
from flax import struct
from abc import ABC
import jumpy as jp
from brax import math


class State(ABC):
    pass


@struct.dataclass
class Pos(object):
    """
    Positions of some object in a particular frame.

    rotation is a quaternion about the center of mass
    """
    rot: jp.ndarray
    pos: jp.ndarray

    @classmethod
    def zero(cls, shape=()):
        return cls(
            pos=jp.zeros(shape + (3,)),
            rot=jp.tile(jp.array([1., 0., 0., 0]), reps=shape + (1,)))

    def __add__(self, other):
        if isinstance(other, Pos):
            return Pos(
                rot=self.rot + other.rot,
                pos=self.pos + other.pos)
        return (PosVel.zero() + self) + other


@struct.dataclass
class TimeDerivatives(object):
    """
    Velocities of some object with respect to some frame.
    """
    vel: jp.ndarray
    ang: jp.ndarray

    @ classmethod
    def zero(cls, shape=()):
        return cls(
            vel=jp.zeros(shape + (3,)),
            ang=jp.zeros(shape + (3,)))

    def __add__(self, other):
        if isinstance(other, TimeDerivatives):
            return TimeDerivatives(
                vel=self.vel + other.vel,
                ang=self.ang + other.ang)
        return (PosVel.zero() + self) + other


@ struct.dataclass
class PosVel(object):
    rot: jp.ndarray
    pos: jp.ndarray
    vel: jp.ndarray
    ang: jp.ndarray

    @ classmethod
    def zero(cls, shape=()):
        return cls(
            pos=jp.zeros(shape + (3,)),
            rot=jp.tile(jp.array([1., 0., 0., 0]), reps=shape + (1,)),
            vel=jp.zeros(shape + (3,)),
            ang=jp.zeros(shape + (3,)))

    def __add__(self, other):
        if isinstance(other, PosVel):
            return PosVel(
                pos=self.pos + other.pos,
                rot=self.rot + other.rot,
                vel=self.vel + other.vel,
                ang=self.ang + other.ang)
        elif isinstance(other, Pos):
            return PosVel(
                pos=self.pos + other.pos,
                rot=self.rot + other.rot,
                vel=self.vel,
                ang=self.ang)
        elif isinstance(other, TimeDerivatives):
            return PosVel(
                pos=self.pos,
                rot=self.rot,
                vel=self.vel + other.vel,
                ang=self.ang + other.ang)
        raise NotImplementedError

    def to_world(self, rpos: jp.ndarray) -> Tuple[jp.ndarray, jp.ndarray]:
        """Returns world information about a point relative to a part.
        Args:
        rpos: Point relative to center of mass of part.
        Returns:
        A 2-tuple containing:
            * World-space coordinates of rpos
            * World-space velocity of rpos
        """
        rpos_off = math.rotate(rpos, self.rot)
        rvel = jp.cross(self.ang, rpos_off)
        return (self.pos + rpos_off, self.vel + rvel)

    def world_velocity(self, pos: jp.ndarray) -> jp.ndarray:
        """Returns the velocity of the point on a rigidbody in world space.
        Args:
        pos: World space position which to use for velocity calculation.
        """
        return self.vel + jp.cross(self.ang, pos - self.pos)