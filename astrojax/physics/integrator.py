from abc import ABC
from typing import Optional

import jax
import jumpy as jp
from astrojax.state.state import Pos, PosVel, TimeDerivatives
from brax import pytree


def ang_to_quat(ang: jp.ndarray) -> jp.ndarray:
    return jp.array([0, ang[0], ang[1], ang[2]])


def quat_mul(u: jp.ndarray, v: jp.ndarray) -> jp.ndarray:
    """Multiplies two quaternions.
    Args:
      u: (4,) quaternion (w,x,y,z)
      v: (4,) quaternion (w,x,y,z)
    Returns:
      A quaternion u * v.
    """
    return jp.array([
        u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
        u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
        u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
        u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
    ])


@pytree.register
class Integrator:
    """
    We update the state of the system by linearing the update equations
    """

    def __init__(self, dt: float, num_bodies: int):
        self.dt = dt
        self.pos_mask = 1. * jp.logical_not(jp.zeros((num_bodies, 3,)))
        self.rot_mask = 1. * jp.logical_not(jp.zeros((num_bodies, 3,)))
        self.quat_mask = 1. * jp.logical_not(jp.zeros((num_bodies, 4,)))

    def kinetic(self, qp: PosVel) -> PosVel:

        @jp.vmap
        def op(qp, pos_mask, rot_mask) -> PosVel:
            pos = qp.pos + qp.vel * self.dt * pos_mask
            rot_at_ang_quat = ang_to_quat(qp.ang * rot_mask) * 0.5 * self.dt
            rot = qp.rot + quat_mul(rot_at_ang_quat, qp.rot)
            rot = rot / jp.norm(rot)
            return PosVel(rot, pos, qp.vel, qp.ang)
        return op(qp, self.pos_mask, self.rot_mask)

    def update(self,
               qp: PosVel,
               acc_p: Optional[TimeDerivatives] = None,
               vel_p: Optional[TimeDerivatives] = None,
               pos_q: Optional[Pos] = None) -> PosVel:
        """Performs an arg dependent integrator step.
        Args:
          qp: State data to be integrated
          acc_p: Acceleration level updates to apply to qp
          vel_p: Velocity level updates to apply to qp
          pos_q: Position level updates to apply to qp
        Returns:
          State data advanced by one potential integration step.
        """

        @jp.vmap
        def op_acc(qp, dp) -> PosVel:
            vel = qp.vel
            vel += dp.vel * self.dt
            ang = qp.ang
            ang += dp.ang * self.dt
            return PosVel(pos=qp.pos, rot=qp.rot, vel=vel, ang=ang)

        @jp.vmap
        def op_vel(qp, dp) -> PosVel:
            vel = (qp.vel + dp.vel)
            ang = (qp.ang + dp.ang)
            return PosVel(pos=qp.pos, rot=qp.rot, vel=vel, ang=ang)

        @jp.vmap
        def op_pos(qp, dq) -> PosVel:
            qp = PosVel(
                pos=qp.pos + dq.pos,
                rot=qp.rot + dq.rot,
                ang=qp.ang,
                vel=qp.vel)
            return qp

        if acc_p:
            return op_acc(qp, acc_p)
        elif vel_p:
            return op_vel(qp, vel_p)
        elif pos_q:
            return op_pos(qp, pos_q)
        else:
            return qp
