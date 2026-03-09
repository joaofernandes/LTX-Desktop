"""GGUF state dict loader for ltx_core model builders.

Loads quantized GGUF checkpoints (e.g. Q3_K_M) and dequantizes tensors to
bfloat16 so they can be consumed by the standard ltx_core model pipeline.

Also provides:
- MultiSourceStateDictLoader: dispatches to GGUF or safetensors loader
  based on file extension, allowing mixed model_path tuples.
- GGUF_AV_GEMMA_TEXT_ENCODER_KEY_OPS: SDOps variant that handles the
  GGUF key naming convention (no model.diffusion_model. prefix) and
  the FP8 Gemma key prefix (model.* instead of language_model.model.*).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _build_gguf_text_encoder_key_ops():
    """Build SDOps for the text encoder when loading from GGUF + FP8 Gemma."""
    from ltx_core.loader.sd_ops import (
        ContentMatching,
        ContentReplacement,
        KeyValueOperationResult,
        SDKeyValueOperation,
        SDOps,
    )

    return SDOps(
        name="GGUF_AV_GEMMA_TEXT_ENCODER_KEY_OPS",
        mapping=(
            # --- From connector.safetensors ---
            ContentMatching(prefix="text_embedding_projection.aggregate_embed.", suffix=""),
            ContentReplacement(
                content="text_embedding_projection.aggregate_embed.",
                replacement="feature_extractor.aggregate_embed.",
            ),
            ContentMatching(prefix="text_embedding_projection.video_aggregate_embed.", suffix=""),
            ContentReplacement(
                content="text_embedding_projection.video_aggregate_embed.",
                replacement="feature_extractor.video_aggregate_embed.",
            ),
            ContentMatching(prefix="text_embedding_projection.audio_aggregate_embed.", suffix=""),
            ContentReplacement(
                content="text_embedding_projection.audio_aggregate_embed.",
                replacement="feature_extractor.audio_aggregate_embed.",
            ),
            # --- From GGUF (no model.diffusion_model. prefix) ---
            ContentMatching(prefix="video_embeddings_connector.", suffix=""),
            ContentReplacement(
                content="video_embeddings_connector.",
                replacement="embeddings_processor.video_connector.",
            ),
            ContentMatching(prefix="audio_embeddings_connector.", suffix=""),
            ContentReplacement(
                content="audio_embeddings_connector.",
                replacement="embeddings_processor.audio_connector.",
            ),
            # --- From FP8 Gemma safetensors (model.* prefix, not language_model.model.*) ---
            ContentMatching(prefix="model.", suffix=""),
            ContentReplacement(
                content="model.",
                replacement="model.model.language_model.",
            ),
            # Weight tying: duplicate embed_tokens.weight as lm_head.weight
            SDKeyValueOperation(
                key_matcher=ContentMatching(
                    prefix="model.model.language_model.embed_tokens.weight", suffix=""
                ),
                kv_operation=lambda key, value: [
                    KeyValueOperationResult(key, value),
                    KeyValueOperationResult("model.lm_head.weight", value),
                ],
            ),
        ),
    )


GGUF_AV_GEMMA_TEXT_ENCODER_KEY_OPS = _build_gguf_text_encoder_key_ops()


class GGUFStateDictLoader:
    """Loads a GGUF checkpoint, dequantizes all tensors to bfloat16.

    Implements the StateDictLoader interface expected by
    ltx_core.loader.single_gpu_model_builder.SingleGPUModelBuilder.
    """

    def metadata(self, path: str) -> dict:
        """Read model config from the GGUF metadata field named 'config'."""
        from gguf import GGUFReader

        reader = GGUFReader(path, mode="r")
        field = reader.get_field("config")
        if field is None:
            raise ValueError(f"No 'config' metadata field in GGUF: {path}")
        raw = bytes(field.parts[-1]).decode("utf-8")
        return json.loads(raw)

    def load(
        self,
        path: str | list[str],
        sd_ops=None,
        device: torch.device | None = None,
    ):
        """Load and dequantize tensors from a GGUF file.

        Only the first path in a list is treated as the GGUF; additional
        paths are ignored (use MultiSourceStateDictLoader for mixed lists).
        """
        import gguf.quants as quants
        from gguf import GGUFReader
        from ltx_core.loader.primitives import StateDict

        gguf_path = path[0] if isinstance(path, list) else path
        device = device or torch.device("cpu")

        logger.info("Loading GGUF checkpoint: %s", gguf_path)
        reader = GGUFReader(gguf_path, mode="r")

        sd: dict[str, torch.Tensor] = {}
        total_size = 0

        for tensor in reader.tensors:
            name = tensor.name
            new_name = sd_ops.apply_to_key(name) if sd_ops is not None else name
            if new_name is None:
                continue

            # Dequantize: returns numpy float32 array in PyTorch shape order
            arr = quants.dequantize(tensor.data, tensor.tensor_type)
            value = torch.from_numpy(arr.copy()).to(dtype=torch.bfloat16, device=device)

            if sd_ops is not None:
                pairs = sd_ops.apply_to_key_value(new_name, value)
                for k, v in pairs:
                    sd[k] = v
                    total_size += v.nbytes
            else:
                sd[new_name] = value
                total_size += value.nbytes

        logger.info("Loaded %d tensors from GGUF (%d MB)", len(sd), total_size // 1024 // 1024)
        return StateDict(
            sd=sd,
            device=device,
            size=total_size,
            dtype={v.dtype for v in sd.values()},
        )


class GGUFModelStateDictLoader(GGUFStateDictLoader):
    """GGUFStateDictLoader with metadata() support (mirrors SafetensorsModelStateDictLoader)."""

    pass  # metadata() already implemented in GGUFStateDictLoader


class MultiSourceStateDictLoader:
    """Combines GGUF and safetensors loaders based on file extension.

    Allows model_path tuples that mix .gguf and .safetensors paths.
    Metadata is read from the first path in the list.
    """

    def metadata(self, path: str) -> dict:
        if path.endswith(".gguf"):
            return GGUFStateDictLoader().metadata(path)
        from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader

        return SafetensorsModelStateDictLoader().metadata(path)

    def load(
        self,
        path: str | list[str],
        sd_ops=None,
        device: torch.device | None = None,
    ):
        from ltx_core.loader.primitives import StateDict
        from ltx_core.loader.sft_loader import SafetensorsStateDictLoader

        paths = [path] if isinstance(path, str) else list(path)
        device = device or torch.device("cpu")

        combined_sd: dict[str, torch.Tensor] = {}
        total_size = 0
        all_dtypes: set = set()

        sft_loader = SafetensorsStateDictLoader()
        gguf_loader = GGUFStateDictLoader()

        for p in paths:
            if p.endswith(".gguf"):
                chunk = gguf_loader.load(p, sd_ops=sd_ops, device=device)
            else:
                chunk = sft_loader.load(p, sd_ops=sd_ops, device=device)
            combined_sd.update(chunk.sd)
            total_size += chunk.size
            all_dtypes.update(chunk.dtype)

        return StateDict(
            sd=combined_sd,
            device=device,
            size=total_size,
            dtype=all_dtypes,
        )
