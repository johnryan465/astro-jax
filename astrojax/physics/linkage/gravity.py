from turtle import distance
from typing import Any, List, Tuple
from brax import pytree
from brax import math
from astrojax.physics.bodies import Body
from flax import struct
from astrojax.physics.linkage.base import Linkage, LinkageConfig


from astrojax.state.state import Pos, PosVel, TimeDerivatives
import jumpy as jp


@struct.dataclass
class GravityLinkageConfig:
    parent_name: str
    child_name: str
    parent_idx: int
    child_idx: int
    parent_mass: float
    child_mass: float
    gravity_coeff: float = 6.67430*(1e-11)




@pytree.register
class TwoBodyGravity(Linkage):
    """
    We can model two body gravity similar in style to Braxes spherical spring joint.
    """

    def __init__(self, body: Body, linkages: List[GravityLinkageConfig]):
        joints = []
        self.body_p = jp.take(body, [body.index[j.parent_name] for j in linkages])
        self.body_c = jp.take(body, [body.index[j.child_name] for j in linkages])
        self.mass_p = jp.array([j.parent_mass for j in linkages])
        self.mass_c = jp.array([j.child_mass for j in linkages])
        self.index = {j.name: i for i, j in enumerate(joints)}
        self.g = jp.array([j.gravity_coeff for j in linkages])


    def apply(self, qp: PosVel) -> TimeDerivatives:
        """Returns impulses to apply to the bodies."""
        qp_p = jp.take(qp, self.body_p.idx)
        qp_c = jp.take(qp, self.body_c.idx)
        dp_p, dp_c = jp.vmap(type(self).apply_reduced)(self, qp_p, qp_c)
        # sum together all impulse contributions across parents and children
        body_idx = jp.concatenate((self.body_p.idx, self.body_c.idx))
        dp_vel = jp.concatenate((dp_p.vel, dp_c.vel))  # pytype: disable=attribute-error
        dp_ang = jp.concatenate((dp_p.ang, dp_c.ang))  # pytype: disable=attribute-error
        dp_vel = jp.segment_sum(dp_vel, body_idx, qp.pos.shape[0])
        dp_ang = jp.segment_sum(dp_ang, body_idx, qp.pos.shape[0])

        return TimeDerivatives(vel=dp_vel, ang=dp_ang)

    def apply_reduced(self, qp_p: PosVel, qp_c: PosVel) -> Tuple[TimeDerivatives, TimeDerivatives]:
        """Returns calculated impulses in compressed joint space."""
        dist = qp_p.pos - qp_c.pos
        norm = jp.norm(dist, axis=-1)
        r_squared = jp.square(norm)
        normal = dist / norm

        dp_p = TimeDerivatives(
            vel = -self.mass_c * self.g * normal / r_squared,
            ang = jp.zeros_like(qp_p.ang)
        )
        
        dp_c = TimeDerivatives(
            vel = self.mass_p * self.g *  normal / r_squared,
            ang = jp.zeros_like(qp_c.ang)
        )

        return dp_p, dp_c
