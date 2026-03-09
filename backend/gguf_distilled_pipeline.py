"""GGUF-backed distilled pipeline variants.

GGUFDistilledPipeline     — T2V/I2V (mirrors DistilledPipeline)
GGUFDistilledA2VPipeline  — A2V   (mirrors DistilledA2VPipeline)

Both use GGUFModelLedger instead of ModelLedger so the transformer is
loaded from the Q3_K_M GGUF checkpoint while VAE/audio/text components
are loaded from their separate safetensors files.
"""

from __future__ import annotations

import torch

from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.types import PipelineComponents
from services.a2v_pipeline.distilled_a2v_pipeline import DistilledA2VPipeline


class GGUFDistilledPipeline(DistilledPipeline):
    """Two-stage distilled T2V/I2V pipeline using a GGUF transformer."""

    def __init__(
        self,
        gguf_checkpoint_path: str,
        vae_path: str,
        audio_vae_path: str,
        connector_path: str,
        gemma_root: str,
        spatial_upsampler_path: str,
        loras=None,
        device: torch.device | None = None,
        quantization=None,
    ) -> None:
        from gguf_model_ledger import GGUFModelLedger
        from ltx_pipelines.utils.helpers import get_device

        if device is None:
            device = get_device()

        self.device = device
        self.dtype = torch.bfloat16

        self.model_ledger = GGUFModelLedger(
            dtype=self.dtype,
            device=device,
            gguf_checkpoint_path=gguf_checkpoint_path,
            vae_path=vae_path,
            audio_vae_path=audio_vae_path,
            connector_path=connector_path,
            gemma_root_path=gemma_root,
            spatial_upsampler_path=spatial_upsampler_path,
            loras=loras or [],
            quantization=quantization,
        )

        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )
        # Do NOT call super().__init__() — it would create a second ModelLedger.


class GGUFDistilledA2VPipeline(DistilledA2VPipeline):
    """Two-stage distilled A2V pipeline using a GGUF transformer."""

    def __init__(
        self,
        gguf_checkpoint_path: str,
        vae_path: str,
        audio_vae_path: str,
        connector_path: str,
        gemma_root: str,
        spatial_upsampler_path: str,
        loras=None,
        device: torch.device | None = None,
        quantization=None,
    ) -> None:
        from gguf_model_ledger import GGUFModelLedger
        from ltx_pipelines.utils.helpers import get_device
        from ltx_pipelines.utils.types import PipelineComponents

        if device is None:
            device = get_device()

        self.device = device
        self.dtype = torch.bfloat16

        self.model_ledger = GGUFModelLedger(
            dtype=self.dtype,
            device=device,
            gguf_checkpoint_path=gguf_checkpoint_path,
            vae_path=vae_path,
            audio_vae_path=audio_vae_path,
            connector_path=connector_path,
            gemma_root_path=gemma_root,
            spatial_upsampler_path=spatial_upsampler_path,
            loras=loras or [],
            quantization=quantization,
        )

        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )
        # Do NOT call super().__init__() — it would create a second ModelLedger.
