"""GGUFModelLedger — ModelLedger variant that loads transformer from GGUF.

Each component is sourced from its own file:
  - Transformer:      GGUF checkpoint (Q3_K_M quantized)
  - Video VAE:        ltx-2.3-distilled-video-vae.safetensors
  - Audio VAE:        ltx-2.3-distilled-audio-vae.safetensors
  - Vocoder:          ltx-2.3-distilled-audio-vae.safetensors  (same file)
  - Text encoder:     connector.safetensors + GGUF (connectors) + gemma safetensors
  - Spatial upsampler: ltx-2-spatial-upscaler-x2-1.0.safetensors
"""

from __future__ import annotations

import torch

from ltx_pipelines.utils import ModelLedger


class GGUFModelLedger(ModelLedger):
    """ModelLedger that sources the transformer from a GGUF checkpoint.

    Additional constructor parameters (beyond the base ModelLedger):
      gguf_checkpoint_path: path to the .gguf transformer file
      vae_path:             video VAE safetensors
      audio_vae_path:       audio VAE + vocoder safetensors
      connector_path:       text embedding projection safetensors
    """

    def __init__(
        self,
        dtype: torch.dtype,
        device: torch.device,
        gguf_checkpoint_path: str,
        vae_path: str,
        audio_vae_path: str,
        connector_path: str,
        gemma_root_path: str | None = None,
        spatial_upsampler_path: str | None = None,
        loras=None,
        registry=None,
        quantization=None,
    ):
        # Store GGUF-specific paths before calling super().__init__,
        # which immediately calls build_model_builders().
        self.gguf_checkpoint_path = gguf_checkpoint_path
        self.vae_path = vae_path
        self.audio_vae_path = audio_vae_path
        self.connector_path = connector_path

        super().__init__(
            dtype=dtype,
            device=device,
            # checkpoint_path=None so base __init__ skips its own builder setup;
            # we override build_model_builders() instead.
            checkpoint_path=gguf_checkpoint_path,
            gemma_root_path=gemma_root_path,
            spatial_upsampler_path=spatial_upsampler_path,
            loras=loras,
            registry=registry,
            quantization=quantization,
        )

    def build_model_builders(self) -> None:
        from ltx_core.loader.registry import DummyRegistry
        from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
        from ltx_core.model.audio_vae import (
            AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
            AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
            VOCODER_COMFY_KEYS_FILTER,
            AudioDecoderConfigurator,
            AudioEncoderConfigurator,
            VocoderConfigurator,
        )
        from ltx_core.model.transformer import (
            LTXV_MODEL_COMFY_RENAMING_MAP,
            LTXModelConfigurator,
        )
        from ltx_core.model.upsampler import LatentUpsamplerConfigurator
        from ltx_core.model.video_vae import (
            VAE_DECODER_COMFY_KEYS_FILTER,
            VAE_ENCODER_COMFY_KEYS_FILTER,
            VideoDecoderConfigurator,
            VideoEncoderConfigurator,
        )
        from ltx_core.text_encoders.gemma import (
            GEMMA_MODEL_OPS,
            GemmaTextEncoderConfigurator,
            module_ops_from_gemma_root,
        )
        from ltx_core.utils import find_matching_file

        from gguf_loader import (
            GGUF_AV_GEMMA_TEXT_ENCODER_KEY_OPS,
            GGUFModelStateDictLoader,
            MultiSourceStateDictLoader,
        )

        registry = self.registry or DummyRegistry()

        # --- Transformer: loaded from GGUF ---
        self.transformer_builder = Builder(
            model_path=self.gguf_checkpoint_path,
            model_class_configurator=LTXModelConfigurator,
            model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
            loras=tuple(self.loras),
            registry=registry,
            model_loader=GGUFModelStateDictLoader(),
        )

        # --- Video VAE encoder & decoder ---
        self.vae_decoder_builder = Builder(
            model_path=self.vae_path,
            model_class_configurator=VideoDecoderConfigurator,
            model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
            registry=registry,
        )
        self.vae_encoder_builder = Builder(
            model_path=self.vae_path,
            model_class_configurator=VideoEncoderConfigurator,
            model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
            registry=registry,
        )

        # --- Audio VAE encoder, decoder & vocoder (all in audio_vae_path) ---
        self.audio_encoder_builder = Builder(
            model_path=self.audio_vae_path,
            model_class_configurator=AudioEncoderConfigurator,
            model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
            registry=registry,
        )
        self.audio_decoder_builder = Builder(
            model_path=self.audio_vae_path,
            model_class_configurator=AudioDecoderConfigurator,
            model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
            registry=registry,
        )
        self.vocoder_builder = Builder(
            model_path=self.audio_vae_path,
            model_class_configurator=VocoderConfigurator,
            model_sd_ops=VOCODER_COMFY_KEYS_FILTER,
            registry=registry,
        )

        # --- Text encoder: connector + GGUF connectors + Gemma weights ---
        if self.gemma_root_path is not None:
            module_ops = module_ops_from_gemma_root(self.gemma_root_path)
            model_folder = find_matching_file(self.gemma_root_path, "model*.safetensors").parent
            gemma_weight_paths = [str(p) for p in model_folder.rglob("*.safetensors")]

            # model_path tuple: connector first (text_embedding_projection.*),
            # then GGUF (video/audio_embeddings_connector.*),
            # then Gemma weights (model.* → language_model.model.*)
            text_encoder_paths = (
                self.connector_path,
                self.gguf_checkpoint_path,
                *gemma_weight_paths,
            )

            self.text_encoder_builder = Builder(
                model_path=text_encoder_paths,
                model_class_configurator=GemmaTextEncoderConfigurator,
                model_sd_ops=GGUF_AV_GEMMA_TEXT_ENCODER_KEY_OPS,
                registry=registry,
                module_ops=(GEMMA_MODEL_OPS, *module_ops),
                model_loader=MultiSourceStateDictLoader(),
            )

        # --- Spatial upsampler ---
        if self.spatial_upsampler_path is not None:
            self.upsampler_builder = Builder(
                model_path=self.spatial_upsampler_path,
                model_class_configurator=LatentUpsamplerConfigurator,
                registry=registry,
            )
