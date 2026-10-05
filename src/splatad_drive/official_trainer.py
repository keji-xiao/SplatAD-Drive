"""Local orchestration of the pinned upstream SplatAD trainer; no upstream edits."""
from nerfstudio.engine.trainer import Trainer

from .trainer_checkpoint import ResumableTrainerMixin


class SplatADDriveTrainer(ResumableTrainerMixin, Trainer):
    """Official model with atomic periodic checkpoints and correct resume order."""


def entrypoint():
    import tyro
    from nerfstudio.scripts import train as upstream

    config = tyro.cli(upstream.AnnotatedBaseConfigUnion, description=__doc__)
    if config.method_name != "splatad":
        raise ValueError("This entrypoint supports the pinned SplatAD method only")
    if config.load_config is not None:
        raise ValueError("Use --load-checkpoint with explicit target/config options; --load-config would override the local trainer")
    if config.machine.num_devices != 1 or config.machine.num_machines != 1:
        raise ValueError("This recovery implementation is validated for one training device only")
    if not 1 <= config.steps_per_save <= 2500:
        raise ValueError("steps_per_save must lie in [1, 2500]")
    config._target = SplatADDriveTrainer
    upstream.main(config)

