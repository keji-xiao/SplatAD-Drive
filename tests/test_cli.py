import json
from dataclasses import asdict
from pathlib import Path

import torch
import pytest

from splatad_drive.cli import main,load_reference_scene
from splatad_drive.demo import make_demo,overlay
from splatad_drive.drive import SplatADDrive
from splatad_drive.io import save_json
from splatad_drive.training import load_training_config,build_train_command


def test_official_command_keeps_parser_options_after_subcommand():
    root=Path(__file__).resolve().parents[1]
    config=load_training_config(root/"configs/smoke.yaml")
    argv=build_train_command(config,root,python="custom-python")
    assert argv[0]=="custom-python"
    assert argv.index("--max-num-iterations")<argv.index("pandaset-data")<argv.index("--data")
    assert "--dataset-end-fraction" in argv and "028" in argv
    assert argv[argv.index("--steps-per-save")+1]=="300"


@pytest.mark.parametrize("camera_time", [.5, .51])
def test_scene_json_render_edit_roundtrip(tmp_path, camera_time):
    scene,camera,lidar=make_demo(width=32,height=24)
    camera.timestamp = camera_time
    path=tmp_path/"scene.json"; save_json(path,scene.export())
    loaded=load_reference_scene(path)
    assert torch.equal(loaded.actor_ids,scene.actor_ids)
    torch.testing.assert_close(loaded.means,scene.means)
    requests=tmp_path/"requests.json"; save_json(requests,{"camera":asdict(camera),"lidar":asdict(lidar)})
    edits=tmp_path/"edits.json"; save_json(edits,[{"actor_id":0,"visible":False}])
    assert main(["render","--scene",str(path),"--requests",str(requests),"--edits",str(edits),"--output",str(tmp_path/"renders")])==0
    assert (tmp_path/"renders/rgb.png").is_file()
    assert (tmp_path/"renders/lidar.npz").is_file()
    manifest = json.loads((tmp_path/"renders/render_manifest.json").read_text())
    assert manifest["overlay"]["sensor_time_difference_s"] == pytest.approx(camera_time-lidar.timestamp)
    assert main(["inspect","--scene",str(path),"--output",str(tmp_path/"ply")])==0
    assert (tmp_path/"ply/actor_000.ply").is_file()


def test_overlay_respects_cv_axis_and_valid_mask():
    scene,camera,lidar=make_demo(width=32,height=24)
    rgb=torch.zeros(24,32,3)
    points=torch.tensor([[5.,0.,0.],[-5.,0.,0.]])
    result=overlay(rgb,points,camera,lidar,torch.tensor([True,True]))
    assert result[12,16].sum()>0
    assert (result.sum(-1)>0).sum()==1
    assert torch.equal(overlay(rgb,points,camera,lidar,torch.tensor([False,False])),rgb)
