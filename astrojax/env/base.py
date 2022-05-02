from abc import ABC
from typing import Optional


class Env(ABC):
  """API for driving a brax system for training and inference."""
