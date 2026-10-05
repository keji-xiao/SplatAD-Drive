"""Physical invariants shared by both sensors through the public facade."""
import torch

from splatad_drive import ActorTrajectory,CameraRequest,GaussianScene,LiDARRequest,SplatADDrive


def test_actor_edit_has_same_metric_displacement_in_camera_and_lidar():
    actor=torch.eye(4); actor[2,3]=5.
    scene=GaussianScene([[0.,0.,0.]],[[1.,0.,0.,0.]],[[.3,.2,.1]],[.8],[[.8,.2,.1]],
                        actor_ids=[0],trajectories={0:ActorTrajectory([0.,1.],torch.stack([actor,actor]))})
    drive=SplatADDrive.from_scene(scene)
    camera=CameraRequest(0,.5,torch.eye(4),torch.tensor([[10.,0.,2.5],[0.,10.,2.5],[0.,0.,1.]]),5,5)
    T_world_lidar=torch.eye(4)
    T_world_lidar[:3,:3]=torch.tensor([[0.,-1.,0.],[0.,0.,-1.],[1.,0.,0.]])
    lidar=LiDARRequest(0,.5,T_world_lidar,torch.tensor([0.]),torch.tensor([0.]))
    torch.testing.assert_close(drive.render_camera(camera)["depth"][2,2],torch.tensor(5.))
    torch.testing.assert_close(drive.render_lidar(lidar)["range"][0,0],torch.tensor(5.))
    actor[2,3]=6.
    drive.set_actor_pose(0,.5,actor)
    torch.testing.assert_close(drive.render_camera(camera)["depth"][2,2],torch.tensor(6.))
    torch.testing.assert_close(drive.render_lidar(lidar)["range"][0,0],torch.tensor(6.))
    drive.remove_actor(0)
    assert drive.render_camera(camera)["alpha"].count_nonzero()==0
    assert drive.render_lidar(lidar)["hit_probability"].count_nonzero()==0


def test_ego_shift_preserves_camera_lidar_calibration():
    from splatad_drive import SensorRig
    from splatad_drive.geometry import invert_transform
    T_ego_camera=torch.eye(4); T_ego_camera[2,3]=1.5
    T_ego_lidar=torch.eye(4); T_ego_lidar[0,3]=.3
    rig=SensorRig(T_ego_camera,T_ego_lidar)
    T_world_ego=torch.eye(4); T_world_ego[1,3]=.5
    camera,lidar=rig.sensor_transforms(T_world_ego)
    torch.testing.assert_close(invert_transform(camera)@lidar,invert_transform(T_ego_camera)@T_ego_lidar)
    assert camera[1,3]==.5 and lidar[1,3]==.5
