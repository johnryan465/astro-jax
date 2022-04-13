from flax import struct
from abc import ABC
import jax.numpy as jp


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


@ struct.dataclass
class Vel(object):
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
        if isinstance(other, Vel):
            return Vel(
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
        elif isinstance(other, Vel):
            return PosVel(
                pos=self.pos,
                rot=self.rot,
                vel=self.vel + other.vel,
                ang=self.ang + other.ang)
        raise NotImplementedError
