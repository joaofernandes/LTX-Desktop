"""GGUF-mode pipeline wrappers.

Implements the FastVideoPipeline and A2VPipeline interfaces using
GGUFDistilledPipeline / GGUFDistilledA2VPipeline.

Required environment variables when LTX_USE_GGUF=1:
  LTX_GGUF_CHECKPOINT_PATH   — path to .gguf transformer file
  LTX_GGUF_VAE_PATH          — path to video VAE safetensors
  LTX_GGUF_AUDIO_VAE_PATH    — path to audio VAE safetensors
  LTX_GGUF_CONNECTOR_PATH    — path to text embedding projection safetensors
  LTX_GGUF_UPSAMPLER_PATH    — path to spatial upsampler safetensors
  LTX_GGUF_GEMMA_ROOT        — path to gemma root directory (tokenizer etc.)

The checkpoint_path / upsampler_path / gemma_root args received from
PipelinesHandler are IGNORED — all paths come from env vars so they
can point to the ComfyUI model directory structure.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Final

import torch

from api_types import ImageConditioningInput
from services.ltx_pipeline_common import default_tiling_config, encode_video_output, video_chunks_number
from services.services_utils import AudioOrNone, TilingConfigType, device_supports_fp8


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"Environment variable {name!r} is required when LTX_USE_GGUF=1 but is not set."
        )
    return val


class LTXGGUFFastVideoPipeline:
    """Fast video (T2V/I2V) pipeline backed by a GGUF transformer."""

    pipeline_kind: Final = "fast"

    @staticmethod
    def create(
        checkpoint_path: str,  # ignored — GGUF path comes from env
        gemma_root: str | None,  # ignored — comes from env
        upsampler_path: str,  # ignored — comes from env
        device: torch.device,
    ) -> "LTXGGUFFastVideoPipeline":
        return LTXGGUFFastVideoPipeline(device=device)

    def __init__(self, device: torch.device) -> None:
        from ltx_core.quantization import QuantizationPolicy
        from gguf_distilled_pipeline import GGUFDistilledPipeline

        self.pipeline = GGUFDistilledPipeline(
            gguf_checkpoint_path=_require_env("LTX_GGUF_CHECKPOINT_PATH"),
            vae_path=_require_env("LTX_GGUF_VAE_PATH"),
            audio_vae_path=_require_env("LTX_GGUF_AUDIO_VAE_PATH"),
            connector_path=_require_env("LTX_GGUF_CONNECTOR_PATH"),
            gemma_root=_require_env("LTX_GGUF_GEMMA_ROOT"),
            spatial_upsampler_path=_require_env("LTX_GGUF_UPSAMPLER_PATH"),
            loras=[],
            device=device,
            quantization=QuantizationPolicy.fp8_cast() if device_supports_fp8(device) else None,
        )

    def _run_inference(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        tiling_config: TilingConfigType,
    ) -> tuple[torch.Tensor | Iterator[torch.Tensor], AudioOrNone]:
        from ltx_pipelines.utils.args import ImageConditioningInput as _LtxImageInput

        return self.pipeline(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=[_LtxImageInput(img.path, img.frame_idx, img.strength) for img in images],
            tiling_config=tiling_config,
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        output_path: str,
    ) -> None:
        tiling_config = default_tiling_config()
        video, audio = self._run_inference(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            tiling_config=tiling_config,
        )
        chunks = video_chunks_number(num_frames, tiling_config)
        encode_video_output(
            video=video,
            audio=audio,
            fps=int(frame_rate),
            output_path=output_path,
            video_chunks_number_value=chunks,
        )

    @torch.inference_mode()
    def warmup(self, output_path: str) -> None:
        warmup_frames = 9
        tiling_config = default_tiling_config()
        try:
            video, audio = self._run_inference(
                prompt="test warmup",
                seed=42,
                height=256,
                width=384,
                num_frames=warmup_frames,
                frame_rate=8,
                images=[],
                tiling_config=tiling_config,
            )
            chunks = video_chunks_number(warmup_frames, tiling_config)
            encode_video_output(
                video=video, audio=audio, fps=8, output_path=output_path,
                video_chunks_number_value=chunks,
            )
        finally:
            import os as _os
            if _os.path.exists(output_path):
                _os.unlink(output_path)

    def compile_transformer(self) -> None:
        from typing import cast as _cast
        transformer = self.pipeline.model_ledger.transformer()
        compiled = _cast(
            torch.nn.Module,
            torch.compile(transformer, mode="reduce-overhead", fullgraph=False),
        )
        setattr(self.pipeline.model_ledger, "transformer", lambda: compiled)


class LTXGGUFa2vPipeline:
    """Audio-to-video pipeline backed by a GGUF transformer."""

    @staticmethod
    def create(
        checkpoint_path: str,  # ignored — GGUF path comes from env
        gemma_root: str | None,  # ignored — comes from env
        upsampler_path: str,  # ignored — comes from env
        device: torch.device,
    ) -> "LTXGGUFa2vPipeline":
        return LTXGGUFa2vPipeline(device=device)

    def __init__(self, device: torch.device) -> None:
        from ltx_core.quantization import QuantizationPolicy
        from gguf_distilled_pipeline import GGUFDistilledA2VPipeline

        self.pipeline = GGUFDistilledA2VPipeline(
            gguf_checkpoint_path=_require_env("LTX_GGUF_CHECKPOINT_PATH"),
            vae_path=_require_env("LTX_GGUF_VAE_PATH"),
            audio_vae_path=_require_env("LTX_GGUF_AUDIO_VAE_PATH"),
            connector_path=_require_env("LTX_GGUF_CONNECTOR_PATH"),
            gemma_root=_require_env("LTX_GGUF_GEMMA_ROOT"),
            spatial_upsampler_path=_require_env("LTX_GGUF_UPSAMPLER_PATH"),
            loras=[],
            device=device,
            quantization=QuantizationPolicy.fp8_cast() if device_supports_fp8(device) else None,
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        images: list[ImageConditioningInput],
        audio_path: str,
        audio_start_time: float,
        audio_max_duration: float | None,
        output_path: str,
    ) -> None:
        tiling_config = default_tiling_config()
        video, audio = self.pipeline(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=[(img.path, img.frame_idx, img.strength) for img in images],
            audio_path=audio_path,
            audio_start_time=audio_start_time,
            audio_max_duration=audio_max_duration,
            tiling_config=tiling_config,
        )
        chunks = video_chunks_number(num_frames, tiling_config)
        encode_video_output(
            video=video,
            audio=audio,
            fps=int(frame_rate),
            output_path=output_path,
            video_chunks_number_value=chunks,
        )
