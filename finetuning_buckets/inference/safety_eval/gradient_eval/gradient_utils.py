import torch
from trak.projectors import BasicProjector, CudaProjector


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

    def project_fn(gradient: torch.Tensor, proj_dim: int, seed: int = 0) -> torch.Tensor:
        return fast_jl.project_rademacher_8(gradient, proj_dim, seed, num_sms)

    return project_fn
