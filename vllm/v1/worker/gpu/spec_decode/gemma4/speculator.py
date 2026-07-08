# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 MTP speculator for the V2 GPU model runner."""

from collections import defaultdict

import torch
import torch.nn as nn

from vllm.compilation.backends import set_model_tag
from vllm.config import VllmConfig, get_layers_from_vllm_config, replace
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import (
    EagleSpeculator,
    prepare_eagle_inputs,
)

logger = init_logger(__name__)


class Gemma4Speculator(EagleSpeculator):
    """Gemma4 assistant model support for speculative decoding.

    The assistant is Q-only: draft attention reads K/V from target layers, so
    draft decode steps keep positions and sequence lengths fixed.
    """

    def load_model(self, target_model: nn.Module) -> None:
        target_attn_layer_names = set(
            get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,  # type: ignore[type-abstract]
            ).keys()
        )
        draft_vllm_config = self._create_draft_vllm_config()
        with set_model_tag("eagle_head"):
            self.model = get_model(
                vllm_config=draft_vllm_config,
                model_config=self.speculative_config.draft_model_config,
                load_config=self.speculative_config.draft_load_config,
            )
        self._setup_gemma4_kv_sharing(self.model, target_attn_layer_names)
        self._share_embeddings(self.model, target_model)

        all_attn_layers = set(
            get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,  # type: ignore[type-abstract]
            ).keys()
        )
        self.draft_attn_layer_names = all_attn_layers - target_attn_layer_names

    def _create_draft_vllm_config(self) -> VllmConfig:
        draft_vllm_config = replace(
            self.vllm_config,
            model_config=self.speculative_config.draft_model_config,
        )
        target_backend = self.vllm_config.attention_config.backend
        if target_backend is not None:
            draft_vllm_config = replace(
                draft_vllm_config,
                attention_config=replace(
                    draft_vllm_config.attention_config,
                    backend=target_backend,
                ),
            )
        return draft_vllm_config

    def _setup_gemma4_kv_sharing(
        self,
        model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> None:
        draft_config = self.speculative_config.draft_model_config.hf_config
        draft_text_config = draft_config.get_text_config()
        target_config = self.vllm_config.model_config.hf_config
        target_text_config = target_config.get_text_config()
        target_layer_types = getattr(target_text_config, "layer_types", [])

        if not (hasattr(model, "model") and hasattr(model.model, "layers")):
            return

        target_num_kv_shared = getattr(target_text_config, "num_kv_shared_layers", 0)
        num_non_shared = len(target_layer_types) - target_num_kv_shared
        type_to_target_indices: dict[str, list[int]] = defaultdict(list)
        for idx, layer_type in enumerate(target_layer_types[:num_non_shared]):
            type_to_target_indices[layer_type].append(idx)

        target_prefix = "model.layers"
        for name in target_attn_layer_names:
            if ".layers." in name:
                target_prefix = name.split(".layers.")[0] + ".layers"
                break

        draft_layer_types = getattr(draft_text_config, "layer_types", [])
        for draft_idx, layer in enumerate(model.model.layers):
            if not hasattr(layer, "self_attn"):
                continue
            attn = getattr(layer.self_attn, "attn", None)
            if attn is None:
                continue

            draft_layer_type = (
                draft_layer_types[draft_idx]
                if draft_idx < len(draft_layer_types)
                else "full_attention"
            )
            candidates = type_to_target_indices.get(draft_layer_type, [])
            if not candidates:
                logger.warning(
                    "No target layer of type '%s' for draft layer %d",
                    draft_layer_type,
                    draft_idx,
                )
                continue

            target_idx = candidates[-1]
            attn.kv_sharing_target_layer_name = (
                f"{target_prefix}.{target_idx}.self_attn.attn"
            )

    def _share_embeddings(self, draft_model: nn.Module, target_model: nn.Module) -> None:
        target_language_model = (
            target_model.get_language_model()
            if hasattr(target_model, "get_language_model")
            else target_model
        )
        if get_pp_group().world_size == 1:
            target_embed = getattr(target_language_model.model, "embed_tokens", None)
            if target_embed is not None:
                del draft_model.model.embed_tokens
                draft_model.model.embed_tokens = target_embed

    @torch.inference_mode()
    def run_model(
        self,
        num_tokens: int,
        attn_metadata: dict | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=batch_descriptor,
        ):
            inputs_embeds = None
            if self.supports_mm_inputs:
                mm_embeds, is_mm_embed = mm_inputs or (None, None)
                num_input_tokens = (
                    is_mm_embed.shape[0] if is_mm_embed is not None else num_tokens
                )
                self.inputs_embeds[:num_input_tokens] = self.model.embed_input_ids(
                    self.input_buffers.input_ids[:num_input_tokens],
                    multimodal_embeddings=mm_embeds,
                    is_multimodal=is_mm_embed,
                )
                inputs_embeds = self.inputs_embeds[:num_tokens]

            ret_hidden_states = self.model(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
                hidden_states=self.hidden_states[:num_tokens],
                inputs_embeds=inputs_embeds,
            )
        last_hidden_states, hidden_states = ret_hidden_states
        return last_hidden_states, hidden_states

    def generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        positions = self.input_buffers.positions[:num_reqs]
        idx_mapping = self.idx_mapping[:num_reqs]
        for step in range(1, self.num_speculative_steps):
            last_hidden_states, hidden_states = self.run_model(
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
            )
            last_hidden_states = last_hidden_states[:num_reqs]
            hidden_states = hidden_states[:num_reqs]
            logits = self.model.compute_logits(last_hidden_states)

            draft_tokens = gumbel_sample(
                logits,
                idx_mapping,
                self.temperature,
                self.seeds,
                positions + 1,
                apply_temperature=True,
                processed_logits_out=self.draft_logits[:, step]
                if self.draft_logits is not None
                else None,
            )
            self.draft_tokens[:num_reqs, step] = draft_tokens

            if step < self.num_speculative_steps - 1:
                update_gemma4_inputs(
                    draft_tokens,
                    hidden_states,
                    self.input_buffers,
                    self.hidden_states,
                )

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict,
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        num_tokens = input_batch.num_tokens_after_padding
        num_reqs = input_batch.num_reqs
        max_query_len = input_batch.num_scheduled_tokens.max()

        self.hidden_states[:num_tokens].copy_(last_hidden_states)
        self.temperature.copy_(temperature)
        self.seeds.copy_(seeds)
        self.idx_mapping[:num_reqs].copy_(input_batch.idx_mapping)

        prepare_eagle_inputs(
            self.input_buffers,
            input_batch,
            self.last_token_indices,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            self.max_num_reqs,
        )

        from vllm.v1.worker.gpu.cudagraph_utils import get_uniform_token_count
        from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp

        prefill_batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
            self.prefill_cudagraph_manager,
            num_reqs,
            num_tokens,
            get_uniform_token_count(num_reqs, num_tokens, max_query_len),
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
        )

        if prefill_batch_desc.cg_mode == CUDAGraphMode.FULL:
            self._build_draft_attn_metadata(
                num_reqs=num_reqs,
                num_reqs_padded=prefill_batch_desc.num_reqs or num_reqs,
                num_tokens_padded=prefill_batch_desc.num_tokens,
                max_query_len=self.num_speculative_steps + 1,
            )
            assert self.prefill_cudagraph_manager is not None
            self.prefill_cudagraph_manager.run_fullgraph(prefill_batch_desc)
        else:
            self.prefill(
                num_reqs,
                prefill_batch_desc.num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=prefill_batch_desc.cg_mode,
                mm_inputs=mm_inputs,
            )

        if self.num_speculative_steps == 1:
            return self.draft_tokens[:num_reqs, :1]

        prepare_gemma4_decode(
            self.draft_tokens[:num_reqs, 0],
            self.input_buffers,
            self.max_num_reqs,
        )

        decode_batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
            self.decode_cudagraph_manager,
            num_reqs,
            num_reqs,
            uniform_token_count=1,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
        )

        attn_metadata_updated = None
        slot_mappings_updated = None
        if not (dummy_run and skip_attn_for_dummy_run):
            slot_mappings = self.block_tables.compute_slot_mappings(
                self.idx_mapping[:num_reqs],
                self.input_buffers.query_start_loc[: num_reqs + 1],
                self.input_buffers.positions[:num_reqs],
                decode_batch_desc.num_tokens,
            )
            slot_mappings_updated = build_slot_mappings_by_layer(
                slot_mappings, self.kv_cache_config
            )
            attn_metadata_updated = self._build_draft_attn_metadata(
                num_reqs=num_reqs,
                num_reqs_padded=decode_batch_desc.num_reqs or num_reqs,
                num_tokens_padded=decode_batch_desc.num_tokens,
                max_query_len=1,
            )

        if decode_batch_desc.cg_mode == CUDAGraphMode.FULL:
            assert self.decode_cudagraph_manager is not None
            self.decode_cudagraph_manager.run_fullgraph(decode_batch_desc)
        else:
            self.generate_draft(
                num_reqs,
                decode_batch_desc.num_tokens,
                attn_metadata_updated,
                slot_mappings_updated,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=decode_batch_desc.cg_mode,
            )
        return self.draft_tokens[:num_reqs]


@triton.jit
def _prepare_gemma4_decode_kernel(
    draft_tokens_ptr,
    draft_tokens_stride,
    input_ids_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_reqs = tl.num_programs(0) - 1
    if req_idx == num_reqs:
        for i in range(0, max_num_reqs + 1, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            q = tl.where(block < num_reqs, block, num_reqs)
            mask = block < max_num_reqs + 1
            tl.store(query_start_loc_ptr + block, q, mask=mask)
        for i in range(req_idx, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(seq_lens_ptr + block, 0, mask=mask)
        return

    draft_token = tl.load(draft_tokens_ptr + req_idx * draft_tokens_stride)
    tl.store(input_ids_ptr + req_idx, draft_token)


def prepare_gemma4_decode(
    draft_tokens: torch.Tensor,
    input_buffers: InputBuffers,
    max_num_reqs: int,
):
    num_reqs = draft_tokens.shape[0]
    _prepare_gemma4_decode_kernel[(num_reqs + 1,)](
        draft_tokens,
        draft_tokens.stride(0),
        input_buffers.input_ids,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        max_num_reqs,
        BLOCK_SIZE=1024,
    )


@triton.jit
def _update_gemma4_inputs_kernel(
    input_ids_ptr,
    input_hidden_states_ptr,
    input_hidden_states_stride,
    draft_tokens_ptr,
    output_hidden_states_ptr,
    output_hidden_states_stride,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    draft_token = tl.load(draft_tokens_ptr + req_idx)
    tl.store(input_ids_ptr + req_idx, draft_token)

    for i in range(0, hidden_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < hidden_size
        output_hidden_states = tl.load(
            output_hidden_states_ptr + req_idx * output_hidden_states_stride + block,
            mask=mask,
        )
        tl.store(
            input_hidden_states_ptr + req_idx * input_hidden_states_stride + block,
            output_hidden_states,
            mask=mask,
        )


def update_gemma4_inputs(
    draft_tokens: torch.Tensor,
    output_hidden_states: torch.Tensor,
    input_buffers: InputBuffers,
    hidden_states: torch.Tensor,
):
    num_reqs, hidden_size = output_hidden_states.shape
    _update_gemma4_inputs_kernel[(num_reqs,)](
        input_buffers.input_ids,
        hidden_states,
        hidden_states.stride(0),
        draft_tokens,
        output_hidden_states,
        output_hidden_states.stride(0),
        hidden_size,
        BLOCK_SIZE=1024,
    )
