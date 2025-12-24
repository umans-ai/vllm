# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
UmansCoder: Pixtral Vision Encoder + DeepSeek V3.2 Language Model
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import fields
from functools import cached_property

import torch
from torch import nn
from mistral_common.protocol.instruct.chunk import ImageChunk
from mistral_common.tokens.tokenizers.image import (
    ImageConfig,
    ImageEncoder,
    SpecialImageIDs,
)
from PIL import Image
from transformers import TensorType
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.tokenization_utils_base import TextInput

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargsItems
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalUUIDDict,
    NestedTensors,
)
from vllm.multimodal.parse import ImageProcessorItems, ImageSize, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    MultiModalProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import TokenizerLike, cached_tokenizer_from_config

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .pixtral import (
    PATCH_MERGE,
    PatchMerger,
    PixtralImagePixelInputs,
    VisionEncoderArgs,
    VisionLanguageAdapter,
    VisionTransformer,
)
from .utils import init_vllm_registered_model, maybe_prefix

# Re-use RMSNorm from layernorm module
from vllm.model_executor.layers.layernorm import RMSNorm


class UmansCoderProcessorAdapter:

    def __init__(self, tokenizer: TokenizerLike, image_encoder: ImageEncoder) -> None:
        super().__init__()
        self._tokenizer = tokenizer
        self._image_encoder = image_encoder

    @property
    def tokenizer(self) -> TokenizerLike:
        return self._tokenizer

    @property
    def image_processor(self) -> ImageEncoder:
        return self._image_encoder

    @cached_property
    def image_token_id(self) -> int:
        return self.image_processor.special_ids.img

    @cached_property
    def image_break_id(self) -> int:
        return self.image_processor.special_ids.img_break

    @cached_property
    def image_end_id(self) -> int:
        return self.image_processor.special_ids.img_end

    @cached_property
    def image_size(self) -> int:
        return self.image_processor.mm_config.max_image_size

    @cached_property
    def patch_size(self) -> int:
        return self.image_processor.mm_config.image_patch_size

    def __call__(
        self,
        text: TextInput | list[TextInput] | None = None,
        images: ImageInput | list[ImageInput] | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> Mapping[str, NestedTensors]:
        if text is None:
            text_list: list[str] = []
        elif isinstance(text, list):
            text_list = list(text)
        else:
            text_list = [text]

        if images is None:
            images = []
        if not isinstance(images, list):
            images = [images]

        if not images:
            if not text_list:
                return BatchFeature(dict(input_ids=torch.empty((0, 0), dtype=torch.long)))

            encoded = [
                self.tokenizer.encode(t, add_special_tokens=False)
                for t in text_list
            ]
            max_len = max(len(ids) for ids in encoded) if encoded else 0
            pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
            input_ids = torch.full((len(encoded), max_len), pad_id, dtype=torch.long)
            for i, ids in enumerate(encoded):
                if ids:
                    input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

            return BatchFeature(dict(input_ids=input_ids))

        pixel_values: list[torch.Tensor] = []
        image_sizes: list[tuple[int, int]] = []

        for image in images:
            if hasattr(image, "media"):
                image = image.media

            image_inputs = self.image_processor(ImageChunk(image=image))
            processed_image = torch.tensor(image_inputs.image)
            pixel_values.append(processed_image)
            image_sizes.append((processed_image.shape[1], processed_image.shape[2]))

        encoded = [
            self.tokenizer.encode(t, add_special_tokens=False) for t in text_list
        ]
        max_len = max(len(ids) for ids in encoded) if encoded else 0
        pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        input_ids = torch.full((len(encoded) or 1, max_len), pad_id, dtype=torch.long)
        for i, ids in enumerate(encoded):
            if ids:
                input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

        return BatchFeature(
            dict(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
            )
        )


class UmansCoderProcessingInfo(BaseProcessingInfo):

    def get_tokenizer(self) -> TokenizerLike:
        return cached_tokenizer_from_config(self.ctx.model_config)

    @cached_property
    def _vision_config(self):
        vision_cfg = self.ctx.model_config.hf_config.vision_config
        # vision_config may be a dict or a config object depending on how it was loaded
        if isinstance(vision_cfg, dict):
            return vision_cfg
        return vision_cfg.to_dict() if hasattr(vision_cfg, "to_dict") else vision_cfg

    def _get_vision_value(self, key: str, default=None):
        """Get a value from vision_config, handling both dict and object."""
        vision_cfg = self._vision_config
        if isinstance(vision_cfg, dict):
            return vision_cfg.get(key, default)
        return getattr(vision_cfg, key, default)

    @cached_property
    def _image_encoder(self) -> ImageEncoder:
        hf_config = self.ctx.model_config.hf_config

        # Get image_size from vision_config, with fallback to max_image_size
        image_size = self._get_vision_value("max_image_size")
        if image_size is None:
            image_size = getattr(hf_config, "max_image_size", None)
        if image_size is None:
            image_size = self._get_vision_value("image_size")
        image_size = int(image_size)

        patch_size = int(self._get_vision_value("patch_size"))

        spatial_merge_size = getattr(hf_config, "spatial_merge_size", None)
        if spatial_merge_size is None:
            spatial_merge_size = self._get_vision_value("spatial_merge_size", 1)
        spatial_merge_size = int(spatial_merge_size)

        image_config = ImageConfig(
            image_patch_size=patch_size,
            max_image_size=image_size,
            spatial_merge_size=spatial_merge_size,
        )

        special_ids = SpecialImageIDs(
            img=int(self._get_vision_value("image_token_id")),
            img_break=int(self._get_vision_value("image_break_token_id")),
            img_end=int(self._get_vision_value("image_end_token_id")),
        )

        return ImageEncoder(image_config=image_config, special_ids=special_ids)

    def get_hf_processor(self, **kwargs: object) -> UmansCoderProcessorAdapter:
        return UmansCoderProcessorAdapter(self.get_tokenizer(), self._image_encoder)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: UmansCoderProcessorAdapter | None = None,
    ) -> int:
        if processor is None:
            processor = self.get_hf_processor()

        ncols, nrows = processor.image_processor._image_to_num_tokens(
            Image.new("RGB", (image_width, image_height))
        )
        return ncols * nrows

    def get_image_size_with_most_features(self) -> ImageSize:
        cfg = self._image_encoder.image_config
        return ImageSize(width=cfg.max_image_size, height=cfg.max_image_size)


class UmansCoderDummyInputsBuilder(BaseDummyInputsBuilder[UmansCoderProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        target_width, target_height = self.info.get_image_size_with_most_features()
        image_overrides = mm_options.get("image") if mm_options else None
        return {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                overrides=image_overrides,
            )
        }

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> ProcessorInputs:
        num_images = mm_counts.get("image", 0)
        dummy_mm_data = self.get_dummy_mm_data(seq_len, mm_counts, mm_options)

        processor = self.info.get_hf_processor()
        image_token_id = processor.image_token_id

        dummy_tokens = [image_token_id] * num_images

        return ProcessorInputs(
            prompt=dummy_tokens,
            mm_data=dummy_mm_data,
            tokenization_kwargs={"truncation": False},
        )


class UmansCoderMultiModalProcessor(BaseMultiModalProcessor[UmansCoderProcessingInfo]):

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

        pixel_values = processed_outputs.get("pixel_values")
        if pixel_values is not None:
            image_sizes = processed_outputs.get("image_sizes")
            if isinstance(pixel_values, list) and image_sizes is not None:
                assert len(pixel_values) == len(image_sizes)
                processed_outputs["images"] = [
                    p[:, :h, :w] for p, (h, w) in zip(pixel_values, image_sizes)
                ]
            else:
                processed_outputs["images"] = pixel_values
            processed_outputs.pop("pixel_values", None)

        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: Mapping[str, NestedTensors],
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(images=MultiModalFieldConfig.batched("image"))

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor()
        image_token_id = processor.image_token_id
        image_break_id = processor.image_break_id
        image_end_id = processor.image_end_id

        def get_replacement(item_idx: int):
            images = mm_items.get_items("image", ImageProcessorItems)
            image_size = images.get_image_size(item_idx)

            ncols, nrows = processor.image_processor._image_to_num_tokens(
                Image.new("RGB", (image_size.width, image_size.height))
            )

            tokens = ([image_token_id] * ncols + [image_break_id]) * nrows
            tokens[-1] = image_end_id

            return PromptUpdateDetails.select_token_id(tokens, image_token_id)

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement,
            ),
        ]

    def _cached_apply_hf_processor(
        self,
        prompt: str | list[int],
        mm_data_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> tuple[list[int], MultiModalProcessingInfo, bool]:
        prompt_ids, mm_info, _ = super()._cached_apply_hf_processor(
            prompt=prompt,
            mm_data_items=mm_data_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
            mm_uuids=mm_uuids,
        )
        return prompt_ids, mm_info, False


@MULTIMODAL_REGISTRY.register_processor(
    UmansCoderMultiModalProcessor,
    info=UmansCoderProcessingInfo,
    dummy_inputs=UmansCoderDummyInputsBuilder,
)
class UmansCoderForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<｜img｜>"
        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config

        # Build vision encoder args from vision_config
        vision_config = config.vision_config
        # vision_config may be a dict or a config object
        if isinstance(vision_config, dict):
            vision_config_dict = vision_config
        else:
            vision_config_dict = vision_config.to_dict()
        dataclass_fields = {field.name for field in fields(VisionEncoderArgs)}
        vision_args_dict = {
            key: value
            for key, value in vision_config_dict.items()
            if key in dataclass_fields
        }
        self.vision_args = VisionEncoderArgs(**vision_args_dict)

        # Initialize DeepSeek V3.2 language model
        # Uses flat config (hf_config itself has all DeepSeek fields at top level)
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config,  # flat config with DeepSeek fields
            architectures=["DeepseekV3ForCausalLM"],
            prefix=maybe_prefix(prefix, "language_model"),
        )

        # Initialize vision components (from Pixtral)
        if multimodal_config.get_limit_per_prompt("image"):
            self.vision_encoder = VisionTransformer(self.vision_args)
            self.pre_mm_projector_norm = (
                RMSNorm(self.vision_args.hidden_size, eps=1e-5)
                if self.vision_args.add_pre_mm_projector_layer_norm
                else None
            )
            self.patch_merger = (
                PatchMerger(
                    vision_encoder_dim=self.vision_args.hidden_size,
                    spatial_merge_size=self.vision_args.spatial_merge_size,
                    use_mlp_bias=False,
                )
                if self.vision_args.mm_projector_id == PATCH_MERGE
                else None
            )
            # Use hidden_size from top-level config (DeepSeek LM hidden size)
            self.vision_language_adapter = VisionLanguageAdapter(
                self.vision_args, dim=config.hidden_size
            )
        else:
            self.vision_encoder = None
            self.pre_mm_projector_norm = None
            self.patch_merger = None
            self.vision_language_adapter = None

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> PixtralImagePixelInputs | None:
        images = kwargs.pop("images", None)
        if images is None:
            return None

        return PixtralImagePixelInputs(
            type="pixel_values",
            images=images,
        )

    def _process_image_input(
        self,
        image_input: PixtralImagePixelInputs,
    ) -> tuple[torch.Tensor, ...]:
        assert (
            self.vision_encoder is not None and self.vision_language_adapter is not None
        )

        images = image_input["images"]
        image_features = self.vision_encoder(images)
        feature_sizes = [image_feature.shape[0] for image_feature in image_features]
        image_features = torch.cat(image_features)
        if self.pre_mm_projector_norm is not None:
            image_features = self.pre_mm_projector_norm(image_features)
        if self.patch_merger is not None:
            patch_size = self.vision_args.patch_size
            spatial_merge_size_square = self.vision_args.spatial_merge_size**2
            img_patch_dims = [
                (img.shape[1] // patch_size, img.shape[2] // patch_size)
                for img in images
            ]
            feature_sizes = [
                feature_size // spatial_merge_size_square
                for feature_size in feature_sizes
            ]
            image_features = self.patch_merger(
                image_features, image_sizes=img_patch_dims
            )
        image_embeds = self.vision_language_adapter(image_features)
        image_embeds = torch.split(image_embeds, feature_sizes)
        return image_embeds

    def get_language_model(self) -> nn.Module:
        return self.language_model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        return self._process_image_input(image_input)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Run forward pass for UmansCoder."""
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load weights for vision components and language model."""

        def is_vision_encoder_weights(weight: tuple[str, torch.Tensor]):
            return weight[0].startswith("vision_encoder")

        def is_vision_lang_adapter_weights(weight: tuple[str, torch.Tensor]):
            return weight[0].startswith("vision_language_adapter")

        def is_patch_merger(weight: tuple[str, torch.Tensor]):
            return weight[0].startswith("patch_merger")

        def is_pre_mm_projector_norm(weight: tuple[str, torch.Tensor]):
            return weight[0].startswith("pre_mm_projector_norm")

        # Get references to parameters for direct loading
        vision_encoder_dict = (
            dict(self.vision_encoder.named_parameters())
            if self.vision_encoder is not None
            else {}
        )
        patch_merger_dict = (
            dict(self.patch_merger.named_parameters())
            if self.patch_merger is not None
            else {}
        )
        pre_mm_projector_norm_dict = (
            dict(self.pre_mm_projector_norm.named_parameters())
            if self.pre_mm_projector_norm is not None
            else {}
        )
        vision_lang_adapter_dict = (
            dict(self.vision_language_adapter.named_parameters())
            if self.vision_language_adapter is not None
            else {}
        )

        def llm_weights_generator():
            # Single pass over weights
            for name, w in weights:
                if is_vision_encoder_weights((name, w)):
                    if self.vision_encoder is None:
                        continue
                    # Load vision encoder weights directly
                    trimmed_name = ".".join(name.split(".")[1:])
                    param = vision_encoder_dict[trimmed_name]
                    with torch.no_grad():
                        default_weight_loader(param, w)
                elif is_patch_merger((name, w)):
                    if self.patch_merger is None:
                        continue
                    # Load vision patch merger weights directly
                    trimmed_name = ".".join(name.split(".")[1:])
                    param = patch_merger_dict[trimmed_name]
                    with torch.no_grad():
                        default_weight_loader(param, w)
                elif is_pre_mm_projector_norm((name, w)):
                    if self.pre_mm_projector_norm is None:
                        continue
                    # Load vision pre_mm_projector_norm weights directly
                    trimmed_name = ".".join(name.split(".")[1:])
                    param = pre_mm_projector_norm_dict[trimmed_name]
                    with torch.no_grad():
                        default_weight_loader(param, w)
                elif is_vision_lang_adapter_weights((name, w)):
                    if self.vision_language_adapter is None:
                        continue
                    # Load vision-language adapter weights directly
                    trimmed_name = ".".join(name.split(".")[1:])
                    param = vision_lang_adapter_dict[trimmed_name]
                    with torch.no_grad():
                        default_weight_loader(param, w)
                else:
                    # LLM weights: yield them to be loaded
                    # by language_model.load_weights
                    yield (name, w)

        # Now we call the language model load with the generator
        self.language_model.load_weights(llm_weights_generator())
