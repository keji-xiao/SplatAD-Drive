"""SplatAD-Drive public API; official dependencies are loaded on demand."""
from .scene import GaussianScene, SceneState, ActorTrajectory
from .camera_renderer import CameraRequest
from .lidar_renderer import LiDARRequest
from .drive import SplatADDrive
from .rig import SensorRig

__all__ = ["SplatADDrive", "CameraRequest", "LiDARRequest", "GaussianScene", "SceneState", "ActorTrajectory", "SensorRig"]
__version__ = "0.1.0"
