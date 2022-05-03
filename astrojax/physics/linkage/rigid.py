from abc import abstractmethod
from astrojax.physics.linkage.base import Linkage, LinkageConfig
from astrojax.state.state import PosVel, TimeDerivatives


class RigidLinkage(Linkage):
    """
    A rigid linkage between two bodies. The bodies are linked by point on the parent body and a point on the child body, which constrains the relative motion between the two bodies.    
    """
    
    @abstractmethod
    def apply(self, state: PosVel) -> TimeDerivatives:
        """
        Returns the time derivatives of the parent and child bodies required to maintain a rigid linkage.
        """
