"""Keep sensor calibration fixed while moving the vehicle reference frame."""
from dataclasses import dataclass, replace
import torch
from .geometry import validate_transform


@dataclass
class SensorRig:
    T_ego_camera: torch.Tensor
    T_ego_lidar: torch.Tensor

    def sensor_transforms(self, T_world_ego):
        T_world_ego = torch.as_tensor(T_world_ego)
        T_ego_camera = torch.as_tensor(self.T_ego_camera).to(T_world_ego)
        T_ego_lidar = torch.as_tensor(self.T_ego_lidar).to(T_world_ego)
        for name,value in (("T_world_ego",T_world_ego),("T_ego_camera",T_ego_camera),("T_ego_lidar",T_ego_lidar)):
            validate_transform(value,name)
        return T_world_ego @ T_ego_camera,T_world_ego @ T_ego_lidar

    def requests(self, T_world_ego, camera_request, lidar_request, *, timestamp):
        T_world_camera,T_world_lidar=self.sensor_transforms(T_world_ego)
        return (replace(camera_request,T_world_camera=T_world_camera,timestamp=timestamp),
                replace(lidar_request,T_world_lidar=T_world_lidar,timestamp=timestamp))
