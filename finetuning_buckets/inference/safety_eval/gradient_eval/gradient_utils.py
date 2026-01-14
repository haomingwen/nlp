from typing import Callable, Optional

import torch
from trak.projectors import BasicProjector, CudaProjector, ProjectionType


def get_trak_projector(device: torch.device = torch.device("cuda:0")):
    """Select CUDA or basic projector depending on fast_jl availability."""
    try:
        num_sms = torch.cuda.get_device_properties(device.index).multi_processor_count
        import fast_jl

        # test run to catch at init time if projection goes through
        fast_jl.project_rademacher_8(torch.zeros(8, 1_000, device=device), 512, 0, num_sms)
        projector = CudaProjector
        print("Using CudaProjector")
    except Exception:
        projector = BasicProjector
        print("Using BasicProjector")
    return projector


def get_fastjl_projector(device: torch.device = torch.device("cuda:0")):
    """Return a fast_jl-backed projector callable."""
    try:
        import fast_jl
    except Exception as exc:
        raise RuntimeError("fast_jl is required for fastjl projection.") from exc

    num_sms = torch.cuda.get_device_properties(device.index).multi_processor_count

    def project_fn(vec: torch.Tensor, proj_dim: int, seed: int = 0) -> torch.Tensor:
        return fast_jl.project_rademacher_8(vec, proj_dim, seed, num_sms)

    return project_fn


def project(
    vec: torch.Tensor,
    projector_cls: Optional[CudaProjector],
    proj_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    block_size: int,
    projector_type: str = "trak",
    fastjl_projector: Optional[Callable[[torch.Tensor, int, int], torch.Tensor]] = None,
) -> torch.Tensor:
    """Project high-dimensional matrix to lower dimension with TRAK projector.

    vec: (num_batches, num_params)
    returns: (num_batches, proj_dim) on CPU
    """
    if projector_type == "fastjl":
        if fastjl_projector is None:
            raise ValueError("fastjl_projector is required when projector_type='fastjl'")
        vec = vec.to(device=device, dtype=dtype)
        projected = fastjl_projector(vec, proj_dim, 0)
        return projected.cpu()

    if projector_cls is None:
        raise ValueError("projector_cls is required when projector_type='trak'")
    proj = projector_cls(
        grad_dim=vec.shape[-1],
        proj_dim=proj_dim,
        seed=0,
        proj_type=ProjectionType.rademacher,
        device=device,
        dtype=dtype,
        block_size=block_size,
        max_batch_size=8,
    )
    vec = vec.to(device=device, dtype=dtype)
    projected = proj.project(vec, model_id=0)
    return projected.cpu()
