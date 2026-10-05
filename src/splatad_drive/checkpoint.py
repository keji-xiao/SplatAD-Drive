"""Restricted loading for official checkpoints containing NumPy scalar state."""
from contextlib import contextmanager
import numpy as np
import torch


@contextmanager
def numpy_scalar_checkpoint_context():
    """Allow NumPy scalars used by official schedulers, not arbitrary objects."""
    allowed = [np.core.multiarray.scalar, np.dtype]
    allowed += [type(np.dtype(dtype)) for dtype in (np.float32, np.float64, np.int32, np.int64)]
    if hasattr(torch.serialization, "safe_globals"):
        with torch.serialization.safe_globals(allowed):
            yield
    else:
        yield  # Older supported torch uses the upstream trusted-file loader.
