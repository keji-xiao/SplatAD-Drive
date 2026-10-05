"""CPU-only recovery tests; no official model import or CUDA context is needed."""
from copy import deepcopy
import pickle
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from splatad_drive.trainer_checkpoint import (
    ResumableTrainerMixin,
    atomic_checkpoint_save,
    restricted_checkpoint_load,
)
from splatad_drive.training import build_train_command, load_training_config


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    """Fail if recovery tests accidentally touch CUDA even on a GPU workstation."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def forbidden(*args, **kwargs):
        pytest.fail("CPU recovery test attempted to access CUDA")

    monkeypatch.setattr(torch.cuda, "get_rng_state_all", forbidden)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", forbidden)


class TinyPipeline(torch.nn.Module):
    def __init__(self, count=2):
        super().__init__()
        self.means = torch.nn.Parameter(torch.randn(count, 3))
        self.datamanager = SimpleNamespace(
            train_unseen_cameras=[0, 1, 2], train_unseen_lidars=[0, 1],
            train_dataset=list(range(3)), train_lidar_dataset=list(range(2)),
            eval_unseen_cameras=[1, 0], eval_unseen_lidars=[0],
            eval_dataset=list(range(2)), eval_lidar_dataset=list(range(1)),
        )
        self.loaded_at_step = None

    def load_pipeline(self, state, step):
        # Official SplatAD similarly re-registers parameters after densification.
        self.means = torch.nn.Parameter(state["means"].clone())
        self.loaded_at_step = step


class TinyOptimizers:
    def __init__(self, pipeline):
        self.optimizers = {"means": torch.optim.Adam([pipeline.means], lr=0.05)}
        self.schedulers = {"means": torch.optim.lr_scheduler.ExponentialLR(self.optimizers["means"], gamma=0.9)}

    def load_optimizers(self, state):
        for key, values in state.items():
            self.optimizers[key].load_state_dict(values)

    def load_schedulers(self, state):
        for key, values in state.items():
            self.schedulers[key].load_state_dict(values)


class TinyScaler:
    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        assert state == {}


class PinnedSetupOrder:
    """Matches the relevant upstream order, but uses real small CPU Adam states."""
    def __init__(self, directory, target, checkpoint=None, initial_count=2):
        self.config = SimpleNamespace(
            max_num_iterations=target, load_checkpoint=checkpoint, load_dir=None,
            load_step=None, training_start_step=None, load_optimizer=True,
            load_scheduler=True, save_only_latest_checkpoint=True,
        )
        self.pipeline = TinyPipeline(initial_count)
        self.grad_scaler = TinyScaler()
        self.checkpoint_dir = directory
        self._start_step = 0
        self.local_rank = 0
        self.executed_steps = []
        self.fail_at_step = None

    def setup(self):
        self._load_checkpoint()
        # self.optimizers deliberately does not exist before this line.
        self.optimizers = self.setup_optimizers()

    def setup_optimizers(self):
        return TinyOptimizers(self.pipeline)

    def train(self):
        for step in range(self._start_step, self._start_step + self.config.max_num_iterations):
            if step == self.fail_at_step:
                raise RuntimeError("injected iteration failure")
            self.executed_steps.append(step)
            queue = self.pipeline.datamanager.train_unseen_cameras
            if not queue:
                queue.extend(range(3))
            selected = queue.pop(random.randrange(len(queue)))
            target = torch.randn_like(self.pipeline.means) + float(np.random.random()) + selected
            optimizer = self.optimizers.optimizers["means"]
            optimizer.zero_grad()
            (self.pipeline.means - target).square().mean().backward()
            optimizer.step()
            self.optimizers.schedulers["means"].step()


class CpuTrainer(ResumableTrainerMixin, PinnedSetupOrder):
    pass


def seed_cpu(value):
    random.seed(value)
    np.random.seed(value)
    torch.random.default_generator.manual_seed(value)


def trained_fixture(tmp_path, *, target=3):
    seed_cpu(123)
    trainer = CpuTrainer(tmp_path, target)
    trainer.setup()
    trainer.train()
    trainer.save_checkpoint(target - 1)
    return trainer, tmp_path / f"step-{target - 1:09d}.ckpt"


def test_resume_registers_loaded_parameters_before_restoring_adam_and_scheduler(tmp_path):
    source, checkpoint = trained_fixture(tmp_path / "source")
    resumed = CpuTrainer(tmp_path / "resumed", target=5, checkpoint=checkpoint, initial_count=7)
    old_parameter = resumed.pipeline.means
    resumed.setup()
    assert resumed._start_step == 3
    assert resumed.pipeline.loaded_at_step == 2
    assert resumed.pipeline.means.shape == (2, 3)
    assert resumed.pipeline.means is not old_parameter
    optimizer = resumed.optimizers.optimizers["means"]
    assert optimizer.param_groups[0]["params"][0] is resumed.pipeline.means
    assert optimizer.state[resumed.pipeline.means]["step"] == 3
    assert resumed.optimizers.schedulers["means"].last_epoch == 3
    source_state = source.optimizers.optimizers["means"].state[source.pipeline.means]
    torch.testing.assert_close(optimizer.state[resumed.pipeline.means]["exp_avg"], source_state["exp_avg"])
    resumed.train()
    assert resumed.executed_steps == [3, 4]
    assert resumed.config.max_num_iterations == 5
    assert resumed.optimizers.schedulers["means"].last_epoch == 5


def test_resume_matches_uninterrupted_cpu_updates_rng_and_sampler_order(tmp_path):
    seed_cpu(404)
    uninterrupted = CpuTrainer(tmp_path / "continuous", target=6)
    uninterrupted.setup()
    uninterrupted.train()
    expected_means = uninterrupted.pipeline.means.detach().clone()
    expected_queue = deepcopy(uninterrupted.pipeline.datamanager.train_unseen_cameras)
    expected_next = (random.random(), float(np.random.random()), torch.rand(4))

    seed_cpu(404)
    first = CpuTrainer(tmp_path / "part1", target=3)
    first.setup()
    first.train()
    first.save_checkpoint(2)
    # Construction consumes RNG; restoration must happen only after setup.
    second = CpuTrainer(tmp_path / "part2", target=6, checkpoint=tmp_path / "part1/step-000000002.ckpt")
    second.setup()
    torch.rand(10)
    np.random.random(10)
    random.random()
    second.train()
    assert second.executed_steps == [3, 4, 5]
    torch.testing.assert_close(second.pipeline.means, expected_means, rtol=0, atol=0)
    assert second.pipeline.datamanager.train_unseen_cameras == expected_queue
    assert random.random() == expected_next[0]
    assert float(np.random.random()) == expected_next[1]
    torch.testing.assert_close(torch.rand(4), expected_next[2], rtol=0, atol=0)


def test_eval_sampling_queues_and_resume_provenance_are_restored(tmp_path):
    original, checkpoint = trained_fixture(tmp_path / "source")
    state = restricted_checkpoint_load(checkpoint)
    assert state["splatad_drive"]["sampler_queues"]["eval_unseen_cameras"] == [1, 0]
    assert state["splatad_drive"]["sampler_queues"]["eval_unseen_lidars"] == [0]
    source_bytes = checkpoint.read_bytes()
    resumed = CpuTrainer(tmp_path / "resumed/checkpoints", target=4, checkpoint=checkpoint)
    resumed.base_dir = tmp_path / "resumed"
    resumed.base_dir.mkdir()
    resumed.setup()
    manifest = json.loads((resumed.base_dir / "resume.json").read_text())
    assert manifest["checkpoint"] == str(checkpoint.resolve())
    assert manifest["checkpoint_step"] == 2
    assert manifest["next_step"] == 3
    assert manifest["total_iteration_target"] == 4
    resumed.pipeline.datamanager.eval_unseen_cameras = []
    resumed.pipeline.datamanager.eval_unseen_lidars = []
    resumed.train()
    assert resumed.pipeline.datamanager.eval_unseen_cameras == [1, 0]
    assert resumed.pipeline.datamanager.eval_unseen_lidars == [0]
    resumed.save_checkpoint(3)
    assert checkpoint.read_bytes() == source_bytes


def test_checkpoint_saved_during_resumed_loop_records_total_not_remaining_budget(tmp_path, monkeypatch):
    _, checkpoint = trained_fixture(tmp_path / "source")

    def save_inside_loop(self):
        assert self.config.max_num_iterations == 2  # Pinned loop expects remaining work.
        self.save_checkpoint(4)

    monkeypatch.setattr(PinnedSetupOrder, "train", save_inside_loop)
    resumed = CpuTrainer(tmp_path / "resumed", target=5, checkpoint=checkpoint)
    resumed.setup()
    resumed.train()
    saved = restricted_checkpoint_load(tmp_path / "resumed/step-000000004.ckpt")
    assert saved["splatad_drive"]["total_iteration_target"] == 5
    assert resumed.config.max_num_iterations == 5


def test_legacy_checkpoint_restores_numpy_scalar_scheduler_without_rng_claim(tmp_path, capsys):
    source, checkpoint = trained_fixture(tmp_path)
    state = restricted_checkpoint_load(checkpoint)
    del state["splatad_drive"]
    state["schedulers"]["means"]["_last_lr"] = [np.float64(0.0123)]
    torch.save(state, checkpoint)
    resumed = CpuTrainer(tmp_path / "resumed", target=4, checkpoint=checkpoint)
    resumed.setup()
    assert resumed.optimizers.schedulers["means"].get_last_lr() == [np.float64(0.0123)]
    assert resumed._resume_execution_state is None
    assert "bitwise replay is not promised" in capsys.readouterr().out


def test_restricted_loader_does_not_admit_arbitrary_python_objects(tmp_path):
    path = tmp_path / "untrusted.ckpt"
    torch.save({"step": 2, "pipeline": {"unexpected": TinyScaler()}, "optimizers": {}, "schedulers": {}, "scalers": {}}, path)
    with pytest.raises(pickle.UnpicklingError):
        restricted_checkpoint_load(path)


def test_failed_atomic_save_does_not_replace_existing_file_or_leave_partial(tmp_path, monkeypatch):
    destination = tmp_path / "step-000000002.ckpt"
    atomic_checkpoint_save({"sentinel": "previous valid checkpoint"}, destination)
    previous = destination.read_bytes()

    def fail_during_write(state, stream):
        stream.write(b"partial archive")
        raise OSError("simulated full disk")

    monkeypatch.setattr(torch, "save", fail_during_write)
    with pytest.raises(OSError, match="full disk"):
        atomic_checkpoint_save({"sentinel": "new checkpoint"}, destination)
    assert destination.read_bytes() == previous
    assert not destination.with_name(destination.name + ".tmp").exists()


def test_failed_periodic_save_keeps_previous_checkpoint(tmp_path, monkeypatch):
    trainer, previous = trained_fixture(tmp_path)
    old_bytes = previous.read_bytes()

    def fail(state, path):
        raise OSError("simulated write failure")

    import splatad_drive.trainer_checkpoint as module
    monkeypatch.setattr(module, "atomic_checkpoint_save", fail)
    with pytest.raises(OSError, match="write failure"):
        trainer.save_checkpoint(3)
    assert previous.read_bytes() == old_bytes


def test_periodic_checkpoints_retain_latest_two_and_do_not_delete_unrelated_files(tmp_path):
    trainer, _ = trained_fixture(tmp_path)
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("preserve me")
    trainer.save_checkpoint(3)
    trainer.save_checkpoint(4)
    assert sorted(path.name for path in tmp_path.glob("step-*.ckpt")) == ["step-000000003.ckpt", "step-000000004.ckpt"]
    assert unrelated.read_text() == "preserve me"


def test_total_target_is_restored_when_upstream_iteration_raises(tmp_path):
    _, checkpoint = trained_fixture(tmp_path / "source")
    resumed = CpuTrainer(tmp_path / "resumed", target=5, checkpoint=checkpoint)
    resumed.setup()
    resumed.fail_at_step = 4
    with pytest.raises(RuntimeError, match="iteration failure"):
        resumed.train()
    assert resumed.executed_steps == [3]
    assert resumed.config.max_num_iterations == 5


@pytest.mark.parametrize("attribute,value", [("load_optimizer", False), ("load_scheduler", False), ("training_start_step", 0)])
def test_resume_refuses_partial_optimizer_or_rebased_steps(tmp_path, attribute, value):
    _, checkpoint = trained_fixture(tmp_path / "source")
    resumed = CpuTrainer(tmp_path / "resumed", target=5, checkpoint=checkpoint)
    setattr(resumed.config, attribute, value)
    with pytest.raises(ValueError):
        resumed.setup()


def test_resume_refuses_checkpoint_already_at_target(tmp_path):
    _, checkpoint = trained_fixture(tmp_path / "source")
    resumed = CpuTrainer(tmp_path / "resumed", target=3, checkpoint=checkpoint)
    with pytest.raises(ValueError, match="already completed"):
        resumed.setup()


def test_restoring_sampler_for_incompatible_dataset_fails(tmp_path):
    _, checkpoint = trained_fixture(tmp_path / "source", target=1)
    resumed = CpuTrainer(tmp_path / "resumed", target=2, checkpoint=checkpoint)
    resumed.setup()
    resumed.pipeline.datamanager.train_dataset = []
    with pytest.raises(ValueError, match="outside current dataset"):
        resumed.train()


def test_incompatible_eval_sampler_is_not_silently_reused(tmp_path):
    _, checkpoint = trained_fixture(tmp_path / "source")
    resumed = CpuTrainer(tmp_path / "resumed", target=4, checkpoint=checkpoint)
    resumed.setup()
    resumed.pipeline.datamanager.eval_dataset = []
    with pytest.raises(ValueError, match="eval_unseen_cameras"):
        resumed.train()


def test_periodic_command_is_bounded_and_resume_options_precede_parser():
    root = Path(__file__).resolve().parents[1]
    candidate = load_training_config(root / "configs/baseline.yaml")
    command = build_train_command(candidate, root=root)
    assert command[1] == str(root / "scripts/train_official.py")
    assert command[command.index("--steps-per-save") + 1] == "2000"
    candidate.update(checkpoint_interval=2500, resume_checkpoint="outputs/source/step-000002999.ckpt")
    command = build_train_command(candidate, root=root)
    assert command[command.index("--steps-per-save") + 1] == "2500"
    assert command.index("--load-checkpoint") < command.index("pandaset-data")
    assert command[command.index("--max-num-iterations") + 1] == "30000"


@pytest.mark.parametrize("interval", [0, 2501, True, 1.5])
def test_invalid_periodic_interval_cannot_start_training(interval):
    root = Path(__file__).resolve().parents[1]
    candidate = load_training_config(root / "configs/baseline.yaml")
    candidate["checkpoint_interval"] = interval
    with pytest.raises(ValueError, match="checkpoint_interval"):
        build_train_command(candidate, root=root)
