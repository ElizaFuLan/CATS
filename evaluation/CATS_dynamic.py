import argparse
import torch
import os
import json
import math
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from collections import Counter
import torch.nn as nn

from fastchat.utils import str_to_torch_dtype
from evaluation.eval import run_eval, reorg_answer_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from cats.cats_model import CATSModel

_global_metrics = None
_sv_log_records: List[Dict] = []

_loop_sequence_records: List[Dict] = []
_loop_sequence_run_file: Optional[str] = None

_config = {
    'output_dir': None,
    'model_id': None,
    'bench_name': None,
    'typical_epsilon': 0.0,
    'typical_tau': 0.3,
    'typical_alpha': 0.09,
    'temperature': 0.7,
}

def _get_seq_len_from_kv(kv_cache):
    if kv_cache is None:
        return 0
    if isinstance(kv_cache, list):
        if not kv_cache:
            return 0
        return kv_cache[0][0].shape[2]
    if isinstance(kv_cache, tuple):
        if not kv_cache:
            return 0
        return kv_cache[0][0].shape[2]
    return 0

def _truncate_kv_cache(kv_cache, target_len):
    """Slice KV cache along sequence dimension to target_len."""
    if kv_cache is None:
        return None
    sliced = []
    for key, value in kv_cache:
        if key.shape[2] > target_len:
            sliced.append((key[:, :, :target_len, :], value[:, :, :target_len, :]))
        else:
            sliced.append((key, value))
    if isinstance(kv_cache, tuple):
        return tuple(sliced)
    return sliced

def _truncate_base_past_kv(base_model, target_len, start_layer=0, end_layer=None):
    """Truncate base model KV cache for layers [start_layer, end_layer) to target_len."""
    if base_model.past_key_values is None:
        return
    
    if end_layer is None:
        end_layer = len(base_model.past_key_values)
    
    truncated = []
    for i, (key, value) in enumerate(base_model.past_key_values):
        if start_layer <= i < end_layer and key.shape[2] > target_len:
            truncated.append((key[:, :, :target_len, :], value[:, :, :target_len, :]))
        else:
            truncated.append((key, value))
    base_model.past_key_values = truncated

def _pad_base_past_kv(base_model, target_len, start_layer=0, end_layer=None):
    """Pad base model KV cache for layers [start_layer, end_layer) to target_len by repeating last position."""
    if base_model.past_key_values is None:
        return
    
    if end_layer is None:
        end_layer = len(base_model.past_key_values)
    
    padded = []
    for i, (key, value) in enumerate(base_model.past_key_values):
        if start_layer <= i < end_layer and key.shape[2] < target_len:
            pad_len = target_len - key.shape[2]
            key_last = key[:, :, -1:, :].repeat(1, 1, pad_len, 1)
            value_last = value[:, :, -1:, :].repeat(1, 1, pad_len, 1)
            padded.append((
                torch.cat([key, key_last], dim=2),
                torch.cat([value, value_last], dim=2)
            ))
        else:
            padded.append((key, value))
    base_model.past_key_values = padded

# Multi-branch tree structure builder (supports N correction branches).
def build_tree_structure_from_sv_verification(
    mismatch_list: List[Tuple[int, int]],
    num_draft_tokens: int,
    draft_tokens_list: List[int],
    device: torch.device,
    total_draft_steps: int = 5
):
    num_corrections = len(mismatch_list)
    first_mismatch_pos = mismatch_list[0][0] if num_corrections > 0 else num_draft_tokens

    main_branch_tokens = draft_tokens_list.copy()
    main_branch_len = len(main_branch_tokens)

    branch_tokens = [ct for _, ct in mismatch_list]
    branch_len = num_corrections
    tree_tokens = main_branch_tokens + branch_tokens
    tree_len = len(tree_tokens)

    semantic_position_ids = torch.zeros(tree_len, dtype=torch.long, device=device)
    for i in range(main_branch_len):
        semantic_position_ids[i] = i
    for i, (pos, _) in enumerate(mismatch_list):
        semantic_position_ids[main_branch_len + i] = pos + 1

    tree_attention_mask = torch.zeros(tree_len, tree_len, dtype=torch.bool, device=device)
    for i in range(main_branch_len):
        tree_attention_mask[i, :i + 1] = True
    for i, (pos, _) in enumerate(mismatch_list):
        ci_idx = main_branch_len + i
        tree_attention_mask[ci_idx, :pos + 1] = True
        tree_attention_mask[ci_idx, ci_idx] = True

    tree_indices = torch.arange(tree_len, dtype=torch.long, device=device)

    medusa_choices = []
    for i in range(main_branch_len):
        medusa_choices.append(tuple([0] * i))
    for pos, _ in mismatch_list:
        if pos > 0:
            branch_path = [0] * (pos - 1) + [1]
        else:
            branch_path = [1]
        medusa_choices.append(tuple(branch_path))

    rejected_positions = [pos for pos, _ in mismatch_list]
    corrected_token_indices = [main_branch_len + i for i in range(num_corrections)]
    vk_cache_positions = corrected_token_indices[:]
    vk_semantic_positions = [pos for pos, _ in mismatch_list]

    return {
        'tree_tokens': tree_tokens,
        'tree_attention_mask': tree_attention_mask,
        'tree_position_ids': semantic_position_ids,
        'accepted_tokens': draft_tokens_list[:first_mismatch_pos] if first_mismatch_pos > 0 else [],
        'accepted_len': first_mismatch_pos,
        'tree_len': tree_len,
        'tree_indices': tree_indices,
        'branch_len': branch_len,
        'branch_tokens': branch_tokens,
        'medusa_choices': medusa_choices,
        'main_branch_len': main_branch_len,
        'main_branch_tokens': main_branch_tokens,
        'corrected_token_indices': corrected_token_indices,
        'rejected_positions': rejected_positions,
        'vk_cache_positions': vk_cache_positions,
        'vk_semantic_positions': vk_semantic_positions,
        'num_corrections': num_corrections,
        'mismatch_list': mismatch_list,
        # backward compat (single-correction fallback for downstream code not yet updated)
        'corrected_token_index': corrected_token_indices[0] if num_corrections > 0 else -1,
        'rejected_position': rejected_positions[0] if num_corrections > 0 else -1,
        'vk_cache_position': vk_cache_positions[0] if num_corrections > 0 else -1,
        'vk_semantic_position': vk_semantic_positions[0] if num_corrections > 0 else -1,
    }

def build_dynamic_tree_v2(
    bonus_token: int,
    draft_tokens_list: List[int],
    dm_topk_tokens: List[torch.Tensor],
    dm_topk_logprobs: List[torch.Tensor],
    sv_corrections: List[Tuple[int, int]],
    corr_topk_tokens: List[torch.Tensor],
    corr_topk_logprobs: List[torch.Tensor],
    total_tokens: int,
    device: torch.device,
    sv_score_bonus: float = 2.0,
):
    D = len(draft_tokens_list)
    K = len(dm_topk_tokens[0]) if D > 0 and len(dm_topk_tokens) > 0 else 1

    if D == 0:
        tt = torch.tensor([[bonus_token]], dtype=torch.long, device=device)
        mm = torch.ones((1, 1, 1, 1), dtype=torch.bool, device=device)
        po = torch.zeros(1, dtype=torch.long, device=device)
        ri = torch.tensor([[0]], dtype=torch.long, device=device)
        return tt, mm, po, ri, 0

    budget = (1 + D * K) if total_tokens <= 0 else max(total_tokens, D + 1)

    trunk_cumlp = []
    running = 0.0
    for d in range(D):
        running += float(dm_topk_logprobs[d][0])
        trunk_cumlp.append(running)

    sv_pos_set = {p for p, _ in sv_corrections}

    candidates = []

    # (A) DM siblings
    for d in range(D):
        parent_lp = trunk_cumlp[d - 1] if d > 0 else 0.0
        for k in range(1, K):
            tok = int(dm_topk_tokens[d][k])
            if d in sv_pos_set:
                for p, ct in sv_corrections:
                    if p == d and ct == tok:
                        break
                else:
                    candidates.append((parent_lp + float(dm_topk_logprobs[d][k]),
                                       d, tok, 'dm_sib', None))
                    continue
                continue
            candidates.append((parent_lp + float(dm_topk_logprobs[d][k]),
                               d, tok, 'dm_sib', None))

    corr_scores = []
    for ci, (p, ct) in enumerate(sv_corrections):
        if 0 <= p < D:
            parent_lp = trunk_cumlp[p - 1] if p > 0 else 0.0
            score = parent_lp + sv_score_bonus
            candidates.append((score, p, ct, 'sv_corr', ci))
            corr_scores.append(score)
        else:
            corr_scores.append(0.0)

    for ci in range(len(sv_corrections)):
        if ci >= len(corr_topk_tokens):
            break
        p = sv_corrections[ci][0]
        c_score = corr_scores[ci] if ci < len(corr_scores) else 0.0
        for k in range(len(corr_topk_tokens[ci])):
            tok = int(corr_topk_tokens[ci][k])
            score = c_score + float(corr_topk_logprobs[ci][k])
            candidates.append((score, p, tok, 'corr_cont', ci))

    candidates.sort(key=lambda x: x[0], reverse=True)
    sib_budget = budget - (D + 1)
    selected = candidates[:max(sib_budget, 0)]

    T = D + 1 + len(selected)
    tree_tokens = torch.empty((1, T), dtype=torch.long, device=device)
    tree_tokens[0, 0] = bonus_token
    for d in range(D):
        tree_tokens[0, d + 1] = draft_tokens_list[d]

    sib_types = []
    sib_depths = []
    sib_parent_ci = []
    for i, (sc, depth, tok, typ, pci) in enumerate(selected):
        tree_tokens[0, D + 1 + i] = tok
        sib_types.append(typ)
        sib_depths.append(depth)
        sib_parent_ci.append(pci)

    corr_ci_to_tree_idx = {}
    for i, (typ, pci) in enumerate(zip(sib_types, sib_parent_ci)):
        if typ == 'sv_corr' and pci is not None:
            corr_ci_to_tree_idx[pci] = D + 1 + i

    mask = torch.zeros((T, T), dtype=torch.bool, device=device)
    mask[0, 0] = True
    for d in range(D):
        idx = d + 1
        mask[idx, :idx + 1] = True
    for i in range(len(selected)):
        sib_idx = D + 1 + i
        typ = sib_types[i]
        d = sib_depths[i]
        pci = sib_parent_ci[i]
        if typ in ('dm_sib', 'sv_corr'):
            mask[sib_idx, 0] = True
            if d > 0:
                mask[sib_idx, 1:d + 1] = True
            mask[sib_idx, sib_idx] = True
        elif typ == 'corr_cont':
            mask[sib_idx, 0] = True
            if d > 0:
                mask[sib_idx, 1:d + 1] = True
            if pci is not None and pci in corr_ci_to_tree_idx:
                mask[sib_idx, corr_ci_to_tree_idx[pci]] = True
            mask[sib_idx, sib_idx] = True
    medusa_mask = mask.unsqueeze(0).unsqueeze(0)

    position_offsets = torch.zeros(T, dtype=torch.long, device=device)
    for d in range(D):
        position_offsets[d + 1] = d + 1
    for i in range(len(selected)):
        d = sib_depths[i]
        typ = sib_types[i]
        if typ in ('dm_sib', 'sv_corr'):
            position_offsets[D + 1 + i] = d + 1
        elif typ == 'corr_cont':
            position_offsets[D + 1 + i] = d + 2

    max_path_len = D + 1
    paths = [list(range(D + 1))]

    for i in range(len(selected)):
        sib_idx = D + 1 + i
        typ = sib_types[i]
        d = sib_depths[i]
        pci = sib_parent_ci[i]
        if typ in ('dm_sib', 'sv_corr'):
            path = list(range(d + 1)) + [sib_idx]
            if len(path) > max_path_len:
                max_path_len = len(path)
            paths.append(path)
        elif typ == 'corr_cont':
            if pci is not None and pci in corr_ci_to_tree_idx:
                corr_tree_idx = corr_ci_to_tree_idx[pci]
                path = list(range(d + 1)) + [corr_tree_idx, sib_idx]
            else:
                path = list(range(d + 1)) + [sib_idx]
            if len(path) > max_path_len:
                max_path_len = len(path)
            paths.append(path)

    for j in range(len(paths)):
        while len(paths[j]) < max_path_len:
            paths[j].append(-1)
    retrieve_indices = torch.tensor(paths, dtype=torch.long, device=device)

    return tree_tokens, medusa_mask, position_offsets, retrieve_indices, D


def evaluate_posterior_tree(
    logits: torch.Tensor,
    tree_tokens: torch.Tensor,
    retrieve_indices: torch.Tensor,
    posterior_threshold: float,
    posterior_alpha: float,
    temperature: float,
):
    """EAGLE-style multi-path evaluation (offset convention: logits[i] → tree_tokens[i+1]).

    Returns: best_path (int), accept_length (int, number of accepted draft tokens)
    """
    P, L = retrieve_indices.shape
    safe_ri = retrieve_indices.clamp(min=0)
    path_tokens = tree_tokens[0, safe_ri]
    path_valid = retrieve_indices != -1
    path_logits = logits[0, safe_ri]

    use_typical = posterior_threshold > 0 or posterior_alpha > 0
    if not use_typical:
        preds = path_logits.argmax(dim=-1)
        match = (preds[:, :-1] == path_tokens[:, 1:]) & path_valid[:, 1:]
    else:
        t_safe = max(float(temperature), 1e-8)
        shifted = path_logits[:, :-1, :]
        probs = torch.softmax(shifted / t_safe, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-5)).sum(dim=-1)
        threshold = torch.minimum(
            torch.full_like(entropy, posterior_threshold),
            posterior_alpha * torch.exp(-entropy),
        )
        next_tokens = path_tokens[:, 1:].clamp(min=0)
        draft_prob = torch.gather(probs, -1, next_tokens.unsqueeze(-1)).squeeze(-1)
        match = (draft_prob > threshold) & path_valid[:, 1:]

    cum_match = torch.cumprod(match.int(), dim=1)
    accept_lens = cum_match.sum(dim=1)
    best_path = int(accept_lens.argmax().item())
    return best_path, int(accept_lens[best_path].item())


def _prepare_tm_input_for_tree_attention(
    tree_structure: Dict,
    tree_accepted_hidden: Optional[torch.Tensor],
    tm_hidden_states: torch.Tensor,
    tm_start_index: int,
    global_position_ids: torch.Tensor,
    model,
    device: torch.device,
    tokenizer
) -> Tuple[List[int], torch.Tensor, torch.Tensor, torch.Tensor, int, Dict]:
    tree_info = tree_structure
    accepted_len = tree_info['accepted_len']
    tree_len = tree_info['tree_len']
    main_branch_len = tree_info['main_branch_len']
    rejected_position = tree_info['rejected_position']
    
    if 'tree_hidden_states' in tree_info:
        tree_hidden_states = tree_info['tree_hidden_states']
    else:
        if tree_accepted_hidden is not None and tree_accepted_hidden.shape[1] > 0:
            root_hidden = tree_accepted_hidden[:, -1:, :]
        else:
            root_hidden = tm_hidden_states[:, :1, :] if tm_hidden_states.shape[1] > 0 else None
        
        if root_hidden is not None:
            tree_hidden_states = root_hidden.repeat(1, tree_len, 1)
        else:
            tree_hidden_states = tm_hidden_states[:, :tree_len, :] if tm_hidden_states.shape[1] >= tree_len else tm_hidden_states
    
    tree_accepted_tokens = tree_info.get('accepted_tokens', [])

    semantic_position_ids = tree_info['tree_position_ids']  # [tree_len]
    # tree_position_ids_final = semantic_position_ids + tm_start_index + 1
    tree_position_ids_final = semantic_position_ids + tm_start_index
    tree_position_ids_final = tree_position_ids_final.unsqueeze(0)  # [1, tree_len]
    
    tree_attention_mask = tree_info['tree_attention_mask']
    
    effective_draft_tokens = tree_info['tree_tokens']
    effective_hidden_states = tree_hidden_states
    effective_position_ids = tree_position_ids_final
    
    target_len = len(effective_draft_tokens)
    
    return effective_draft_tokens, effective_hidden_states, effective_position_ids, tree_attention_mask, target_len, {
        'accepted_len': accepted_len,
        'tree_len': tree_len,
        'past_key_values_length_for_tree': tm_start_index,
        'tree_accepted_tokens': tree_accepted_tokens,
        'tree_tokens': tree_info['tree_tokens'],
        'branch_len': tree_info.get('branch_len', 1),
        'main_branch_len': main_branch_len,
        'rejected_position': rejected_position,
        'vk_cache_position': tree_info['vk_cache_position'],
        'vk_semantic_position': tree_info['vk_semantic_position'],
        # multi-branch fields (for Modification 8+)
        'rejected_positions': tree_info.get('rejected_positions', [rejected_position]),
        'corrected_token_indices': tree_info.get('corrected_token_indices', [tree_info.get('corrected_token_index', -1)]),
        'vk_cache_positions': tree_info.get('vk_cache_positions', [tree_info.get('vk_cache_position', -1)]),
        'vk_semantic_positions': tree_info.get('vk_semantic_positions', [tree_info.get('vk_semantic_position', -1)]),
        'num_corrections': tree_info.get('num_corrections', 1),
        'mismatch_list': tree_info.get('mismatch_list', []),
    }


def _prepare_tm_input_for_normal_mode(
    tm_draft_tokens: List[int],
    tm_hidden_states: torch.Tensor,
    tm_start_index: int,
    global_position_ids: torch.Tensor,
    num_draft_tokens: int,
    shallow_accept_mask_for_tm: List[int]
) -> Tuple[List[int], torch.Tensor, torch.Tensor, int]:
    effective_draft_tokens = tm_draft_tokens
    effective_hidden_states = tm_hidden_states
    effective_position_ids = global_position_ids[:, tm_start_index:tm_start_index+len(effective_draft_tokens)]
    target_len = len(effective_draft_tokens)
    
    return effective_draft_tokens, effective_hidden_states, effective_position_ids, target_len

def _forward_tm_with_tree_attention(
    model,
    effective_hidden_states: torch.Tensor,
    effective_position_ids: torch.Tensor,
    tree_attention_mask: torch.Tensor,
    accepted_len: int,
    tree_len: int,
    device: torch.device,
    tm_start_index: int = None
) -> torch.Tensor:
    full_seq_len = effective_hidden_states.shape[1]
    
    full_attention_mask = tree_attention_mask.clone()
    
    if full_attention_mask.shape[0] != full_seq_len or full_attention_mask.shape[1] != full_seq_len:
        full_attention_mask = torch.zeros((full_seq_len, full_seq_len), dtype=torch.bool, device=device)
        min_len = min(tree_attention_mask.shape[0], full_seq_len)
        full_attention_mask[:min_len, :min_len] = tree_attention_mask[:min_len, :min_len]
    
    medusa_mask = full_attention_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, full_seq_len, full_seq_len]

    if tm_start_index is not None:
        past_len = tm_start_index
    else:
        kv_cache_before = 0
        if model.base_model.past_key_values is not None and len(model.base_model.past_key_values) > 0:
            if model.base_model.past_key_values[0][0] is not None:
                kv_cache_before = model.base_model.past_key_values[0][0].shape[2]
        past_len = kv_cache_before
    
    assert medusa_mask.shape == (1, 1, full_seq_len, full_seq_len), \
        f"medusa_mask shape mismatch: expected (1, 1, {full_seq_len}, {full_seq_len}), got {medusa_mask.shape}"
    assert medusa_mask.dtype == torch.bool, \
        f"medusa_mask dtype should be bool, got {medusa_mask.dtype}"
    
    batch_size = 1
    seq_length_with_past = past_len + full_seq_len
    attention_mask_flat = torch.ones((batch_size, seq_length_with_past), dtype=torch.bool, device=device)
    attention_mask = model.base_model.model._prepare_decoder_attention_mask(
        attention_mask_flat, (batch_size, full_seq_len), effective_hidden_states, past_len
    )

    if medusa_mask is not None:
        medusa_len = medusa_mask.size(-1)
        if medusa_len > 0:
            attention_mask[:, :, :, -medusa_len:][
                medusa_mask == 0
            ] = attention_mask.min()

    if not hasattr(model.base_model.model, 'medusa_mask'):
        model.base_model.model.medusa_mask = None
    original_medusa_mask = model.base_model.model.medusa_mask

    try:
        model.base_model.model.medusa_mask = medusa_mask

        _, final_hidden = model.base_model.forward_draft_or_large_model(
            in_features_large=effective_hidden_states,
            position_ids=effective_position_ids
        )
        
    finally:
        model.base_model.model.medusa_mask = original_medusa_mask
    
    return final_hidden


def _build_multi_token_medusa_mask(
    main_branch_len: int,
    mismatch_positions: List[int],
    device: torch.device,
) -> torch.Tensor:
    """Build medusa mask for [draft_bonus/last, c_k1, c_k2, ..., c_kN].
    tree_len = main_branch_len + 1 + N. Returns [(1+N), tree_len]."""
    N = len(mismatch_positions)
    num_new_tokens = 1 + N
    tree_len = main_branch_len + num_new_tokens

    tree_attention_mask = torch.zeros(tree_len, tree_len, dtype=torch.bool, device=device)
    for i in range(main_branch_len):
        tree_attention_mask[i, : i + 1] = True

    draft_idx = main_branch_len
    tree_attention_mask[draft_idx, :main_branch_len] = True
    tree_attention_mask[draft_idx, draft_idx] = True

    for i, pos in enumerate(mismatch_positions):
        ci_idx = main_branch_len + 1 + i
        tree_attention_mask[ci_idx, :pos + 1] = True
        tree_attention_mask[ci_idx, ci_idx] = True

    return tree_attention_mask[-num_new_tokens:, :].clone()  # [1+N, tree_len]


def _forward_tokens_tree_attention_to_draft(
    model,
    batch_tokens: torch.Tensor,
    effective_position_ids: torch.Tensor,
    device: torch.device,
    draft_exit_layer: int,
    round_context_start: int,
    past_len: int,
    main_branch_len: int,
    mismatch_positions: List[int],
) -> torch.Tensor:
    full_seq_len = batch_tokens.shape[1]
    num_new_tokens = 1 + len(mismatch_positions)
    assert full_seq_len == num_new_tokens
    assert effective_position_ids.shape == (1, num_new_tokens)

    medusa_NxT = _build_multi_token_medusa_mask(main_branch_len, mismatch_positions, device)
    tree_len = medusa_NxT.shape[1]
    medusa_mask = medusa_NxT.unsqueeze(0).unsqueeze(0)  # [1, 1, 1+N, tree_len]

    correction_visible_ends = [round_context_start + pos + 1 for pos in mismatch_positions]

    if not hasattr(model.base_model.model, 'medusa_mask'):
        model.base_model.model.medusa_mask = None
    original_medusa_mask = model.base_model.model.medusa_mask

    try:
        model.base_model.model.medusa_mask = medusa_mask
        hidden_states = model.base_model.model.embed_tokens(batch_tokens)

        for layer_idx in range(draft_exit_layer):
            layer = model.base_model.model.layers[layer_idx]
            past_kv = model.base_model.past_key_values[layer_idx]
            current_past_len = past_kv[0].shape[2] if past_kv[0] is not None else 0

            batch_attention_mask_flat = torch.ones((1, current_past_len + full_seq_len), dtype=torch.bool, device=device)
            attention_mask = model.base_model.model._prepare_decoder_attention_mask(
                batch_attention_mask_flat, (1, full_seq_len), hidden_states, current_past_len
            )

            for ci, ve in enumerate(correction_visible_ends):
                row_idx = 1 + ci
                ve_clamped = min(ve, current_past_len)
                if ve_clamped < current_past_len:
                    attention_mask[0, 0, row_idx, ve_clamped:current_past_len] = attention_mask.min()

            medusa_len = medusa_mask.size(-1)
            if medusa_len > 0 and current_past_len + full_seq_len >= medusa_len:
                attention_mask[:, :, :, -medusa_len:][medusa_mask == 0] = attention_mask.min()

            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=effective_position_ids,
                past_key_value=past_kv,
                use_cache=True,
            )
            hidden_states = layer_outputs[0]
            model.base_model.past_key_values[layer_idx] = layer_outputs[1]

        final_hidden = hidden_states
    finally:
        model.base_model.model.medusa_mask = original_medusa_mask
    return final_hidden


def _forward_tokens_tree_attention_draft_to_sv(
    model,
    batch_hidden_draft: torch.Tensor,
    effective_position_ids: torch.Tensor,
    device: torch.device,
    draft_exit_layer: int,
    shallow_exit_layer: int,
    round_context_start: int,
    main_branch_len: int,
    mismatch_positions: List[int],
) -> torch.Tensor:
    """Forward [draft_bonus, c_k1, ..., c_kN] through layers DRAFT_EXIT_LAYER..SHALLOW_EXIT_LAYER-1 with tree attention."""
    full_seq_len = batch_hidden_draft.shape[1]
    num_new_tokens = 1 + len(mismatch_positions)
    assert full_seq_len == num_new_tokens
    assert effective_position_ids.shape == (1, num_new_tokens)

    medusa_NxT = _build_multi_token_medusa_mask(main_branch_len, mismatch_positions, device)
    tree_len = medusa_NxT.shape[1]
    medusa_mask = medusa_NxT.unsqueeze(0).unsqueeze(0)  # [1, 1, 1+N, tree_len]

    correction_visible_ends = [round_context_start + pos + 1 for pos in mismatch_positions]

    if not hasattr(model.base_model.model, 'medusa_mask'):
        model.base_model.model.medusa_mask = None
    original_medusa_mask = model.base_model.model.medusa_mask

    try:
        model.base_model.model.medusa_mask = medusa_mask
        hidden_states = batch_hidden_draft

        for layer_idx in range(draft_exit_layer, shallow_exit_layer):
            layer = model.base_model.model.layers[layer_idx]
            past_kv = model.base_model.past_key_values[layer_idx]
            current_past_len = past_kv[0].shape[2] if past_kv[0] is not None else 0

            batch_attention_mask_flat = torch.ones((1, current_past_len + full_seq_len), dtype=torch.bool, device=device)
            attention_mask = model.base_model.model._prepare_decoder_attention_mask(
                batch_attention_mask_flat, (1, full_seq_len), hidden_states, current_past_len
            )

            for ci, ve in enumerate(correction_visible_ends):
                row_idx = 1 + ci
                ve_clamped = min(ve, current_past_len)
                if ve_clamped < current_past_len:
                    attention_mask[0, 0, row_idx, ve_clamped:current_past_len] = attention_mask.min()

            medusa_len = medusa_mask.size(-1)
            if medusa_len > 0 and current_past_len + full_seq_len >= medusa_len:
                attention_mask[:, :, :, -medusa_len:][medusa_mask == 0] = attention_mask.min()

            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=effective_position_ids,
                past_key_value=past_kv,
                use_cache=True,
            )
            hidden_states = layer_outputs[0]
            model.base_model.past_key_values[layer_idx] = layer_outputs[1]

        final_hidden = hidden_states
    finally:
        model.base_model.model.medusa_mask = original_medusa_mask
    return final_hidden


def _forward_adapter_with_tree_attention(
    adapter,
    batch_hidden: torch.Tensor,
    effective_position_ids: torch.Tensor,
    device: torch.device,
    round_context_start: int,
    main_branch_len: int,
    mismatch_positions: List[int],
    adapter_kv_cache,
) -> Tuple[torch.Tensor, list]:
    """Forward [draft_bonus, c_k1, ..., c_kN] through adapter layers with tree attention.
    Mirrors _forward_tokens_tree_attention_draft_to_sv but operates on adapter.layers.
    Each c_ki only attends to past KV up to its semantic position; no cross-branch attention."""
    full_seq_len = batch_hidden.shape[1]
    num_new_tokens = 1 + len(mismatch_positions)
    assert full_seq_len == num_new_tokens

    medusa_NxT = _build_multi_token_medusa_mask(main_branch_len, mismatch_positions, device)
    medusa_mask = medusa_NxT.unsqueeze(0).unsqueeze(0)  # [1, 1, 1+N, tree_len]

    correction_visible_ends = [round_context_start + pos + 1 for pos in mismatch_positions]

    hidden_states = batch_hidden
    new_kv_cache = []

    for layer_idx, decoder_layer in enumerate(adapter.layers):
        past_kv = adapter_kv_cache[layer_idx] if adapter_kv_cache is not None else None
        current_past_len = past_kv[0].shape[2] if past_kv is not None and past_kv[0] is not None else 0

        batch_attention_mask_flat = torch.ones((1, current_past_len + full_seq_len), dtype=torch.bool, device=device)
        attention_mask = adapter._prepare_decoder_attention_mask(
            batch_attention_mask_flat, (1, full_seq_len), hidden_states, current_past_len
        )

        # Restrict each c_ki to only attend to past KV up to its semantic position.
        for ci, ve in enumerate(correction_visible_ends):
            row_idx = 1 + ci
            ve_clamped = min(ve, current_past_len)
            if ve_clamped < current_past_len:
                attention_mask[0, 0, row_idx, ve_clamped:current_past_len] = attention_mask.min()

        # Apply tree attention mask (disallow cross-branch attention within batch).
        medusa_len = medusa_mask.size(-1)
        if medusa_len > 0 and current_past_len + full_seq_len >= medusa_len:
            attention_mask[:, :, :, -medusa_len:][medusa_mask == 0] = attention_mask.min()

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=effective_position_ids,
            past_key_value=past_kv,
            use_cache=True,
        )
        hidden_states = layer_outputs[0]
        new_kv_cache.append(layer_outputs[1])

    hidden_states = adapter.norm(hidden_states)
    return hidden_states, new_kv_cache


def _forward_tm_with_normal_mode(
    model,
    effective_hidden_states: torch.Tensor,
    effective_position_ids: torch.Tensor,
    effective_draft_tokens: List[int],
    device: torch.device,
    tokenizer
) -> torch.Tensor:
    _, final_hidden = model.base_model.forward_draft_or_large_model(
        in_features_large=effective_hidden_states,
        position_ids=effective_position_ids
    )
    
    return final_hidden

def _sync_kv_cache_for_normal_mode(
    model,
    start_index: int
):
    _truncate_base_past_kv(model.base_model, start_index, 
                          start_layer=0, end_layer=None)

def _handle_tree_attention_cache_after_verification(
    model,
    tree_info_dict: Dict,
    accepted_branch: str,  # 'main' or 'correction_<idx>'
    accepted_count: int,
    tm_start_index: int,
    SHALLOW_EXIT_LAYER: int = 10
) -> int:
    main_branch_len = tree_info_dict['main_branch_len']

    if model.base_model.past_key_values is None:
        return 0

    num_layers = len(model.base_model.past_key_values)

    if accepted_branch == 'main':
        target_len = tm_start_index + accepted_count

        for layer_idx in range(num_layers):
            key, value = model.base_model.past_key_values[layer_idx]
            if key.shape[2] > target_len:
                model.base_model.past_key_values[layer_idx] = (
                    key[:, :, :target_len, :],
                    value[:, :, :target_len, :]
                )

        return target_len

    elif accepted_branch.startswith('correction'):
        corr_idx = int(accepted_branch.split('_')[1]) if '_' in accepted_branch else 0
        rejected_positions = tree_info_dict.get('rejected_positions',
                                                [tree_info_dict.get('rejected_position', 0)])
        actual_rejected_pos = rejected_positions[corr_idx] if corr_idx < len(rejected_positions) else rejected_positions[0]

        vk_src_idx = tm_start_index + main_branch_len + corr_idx
        vk_dst_idx = tm_start_index + actual_rejected_pos + 1

        new_past_key_values = []
        for layer_idx in range(num_layers):
            key, value = model.base_model.past_key_values[layer_idx]
            cache_len = key.shape[2]

            if cache_len <= vk_src_idx:
                new_past_key_values.append((key, value))
                continue

            new_key = key.clone()
            new_value = value.clone()

            new_key[:, :, vk_dst_idx, :] = key[:, :, vk_src_idx, :]
            new_value[:, :, vk_dst_idx, :] = value[:, :, vk_src_idx, :]

            truncate_len = vk_dst_idx + 1
            new_key = new_key[:, :, :truncate_len, :]
            new_value = new_value[:, :, :truncate_len, :]

            new_past_key_values.append((new_key, new_value))

        model.base_model.past_key_values = new_past_key_values

        return vk_dst_idx + 1

    return model.base_model.past_key_values[0][0].shape[2] if model.base_model.past_key_values else 0

def _handle_adapter_kv_cache_after_tree_verification(
    adapter_kv_cache,
    tree_info_dict: Dict,
    accepted_branch: str,
    accepted_count: int,
    tm_start_index: int,
    max_layers: int
):
    if adapter_kv_cache is None:
        return None

    main_branch_len = tree_info_dict['main_branch_len']

    num_layers_in_cache = len(adapter_kv_cache)
    num_layers_to_process = min(num_layers_in_cache, max_layers)

    if accepted_branch == 'main':
        target_len = tm_start_index + accepted_count
        return _truncate_kv_cache(adapter_kv_cache, target_len)

    elif accepted_branch.startswith('correction'):
        corr_idx = int(accepted_branch.split('_')[1]) if '_' in accepted_branch else 0
        rejected_positions = tree_info_dict.get('rejected_positions',
                                                [tree_info_dict.get('rejected_position', 0)])
        actual_rejected_pos = rejected_positions[corr_idx] if corr_idx < len(rejected_positions) else rejected_positions[0]

        vk_src_idx = tm_start_index + main_branch_len + corr_idx
        vk_dst_idx = tm_start_index + actual_rejected_pos + 1
        truncate_len = vk_dst_idx + 1

        new_kv_cache = []
        for layer_idx, (key, value) in enumerate(adapter_kv_cache):
            if layer_idx >= num_layers_to_process:
                if key.shape[2] > truncate_len:
                    new_key = key[:, :, :truncate_len, :].clone()
                    new_value = value[:, :, :truncate_len, :].clone()
                    new_kv_cache.append((new_key, new_value))
                elif key.shape[2] < truncate_len:
                    new_key = key.clone()
                    new_value = value.clone()
                    pad_len = truncate_len - key.shape[2]
                    key_last = new_key[:, :, -1:, :].repeat(1, 1, pad_len, 1)
                    value_last = new_value[:, :, -1:, :].repeat(1, 1, pad_len, 1)
                    new_key = torch.cat([new_key, key_last], dim=2)
                    new_value = torch.cat([new_value, value_last], dim=2)
                    new_kv_cache.append((new_key, new_value))
                else:
                    new_kv_cache.append((key, value))
                continue

            cache_len = key.shape[2]

            if cache_len <= vk_src_idx:
                if cache_len < truncate_len:
                    new_key = key.clone()
                    new_value = value.clone()
                    pad_len = truncate_len - cache_len
                    key_last = new_key[:, :, -1:, :].repeat(1, 1, pad_len, 1)
                    value_last = new_value[:, :, -1:, :].repeat(1, 1, pad_len, 1)
                    new_key = torch.cat([new_key, key_last], dim=2)
                    new_value = torch.cat([new_value, value_last], dim=2)
                    new_kv_cache.append((new_key, new_value))
                elif cache_len > truncate_len:
                    new_key = key[:, :, :truncate_len, :].clone()
                    new_value = value[:, :, :truncate_len, :].clone()
                    new_kv_cache.append((new_key, new_value))
                else:
                    new_kv_cache.append((key, value))
                continue

            new_key = key.clone()
            new_value = value.clone()

            if cache_len <= vk_dst_idx:
                pad_len = vk_dst_idx + 1 - cache_len
                key_last = new_key[:, :, -1:, :].repeat(1, 1, pad_len, 1)
                value_last = new_value[:, :, -1:, :].repeat(1, 1, pad_len, 1)
                new_key = torch.cat([new_key, key_last], dim=2)
                new_value = torch.cat([new_value, value_last], dim=2)

            new_key[:, :, vk_dst_idx, :] = key[:, :, vk_src_idx, :]
            new_value[:, :, vk_dst_idx, :] = value[:, :, vk_src_idx, :]

            new_key = new_key[:, :, :truncate_len, :]
            new_value = new_value[:, :, :truncate_len, :]

            new_kv_cache.append((new_key, new_value))

        if isinstance(adapter_kv_cache, tuple):
            return tuple(new_kv_cache)
        return new_kv_cache

    else:
        target_len = tm_start_index + accepted_count
        return _truncate_kv_cache(adapter_kv_cache, target_len)


@dataclass
class EvaluationMetrics:
    """Per-step evaluation metrics."""
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    
    def update(self, shallow_accept: bool, target_accept: bool):
        if shallow_accept and target_accept:
            self.tp += 1
        elif not shallow_accept and not target_accept:
            self.tn += 1
        elif shallow_accept and not target_accept:
            self.fp += 1
        else:
            self.fn += 1
    
    def compute(self):
        total = self.tp + self.tn + self.fp + self.fn
        if total == 0:
            return 0, 0, 0, 0
        
        accuracy = (self.tp + self.tn) / total
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return accuracy, precision, recall, f1


@dataclass
class GlobalMetrics:
    total_metrics: EvaluationMetrics = field(default_factory=EvaluationMetrics)
    step_metrics: List[EvaluationMetrics] = field(default_factory=list)
    draft_tokens_per_step: List[int] = field(default_factory=list)
    accepted_tokens_per_step: List[int] = field(default_factory=list)
    sv_accepted_tokens_per_step: List[int] = field(default_factory=list)
    
    def add_step(self, step_metric: EvaluationMetrics, num_draft: int, num_accepted: int, sv_accepted: int):
        self.step_metrics.append(step_metric)
        self.draft_tokens_per_step.append(num_draft)
        self.accepted_tokens_per_step.append(num_accepted)
        self.sv_accepted_tokens_per_step.append(sv_accepted)
        
        self.total_metrics.tp += step_metric.tp
        self.total_metrics.tn += step_metric.tn
        self.total_metrics.fp += step_metric.fp
        self.total_metrics.fn += step_metric.fn
    
    def print_summary(self, expected_draft_tokens: int):
        pass

class MultiAdapterCATSModel:
    
    def __init__(self, base_model_path: str, adapter_configs: Dict, args):
        self.adapter_configs = adapter_configs
        self.dtype = str_to_torch_dtype(args.dtype)
        
        max_layer = max(config['layer'] for config in adapter_configs.values())
        self.max_early_stop_layer = max_layer
        
        first_adapter_name = list(adapter_configs.keys())[0]
        first_adapter_config = adapter_configs[first_adapter_name]
        
        temp_cats = CATSModel(
            base_model_path,
            first_adapter_config['path'],
            args,
            EARLY_STOP_LAYER=self.max_early_stop_layer
        )
        
        self.base_model = temp_cats.base_model
        self.AdapterModelClass = type(temp_cats.adapter_model)
        self.first_adapter_name = first_adapter_name
        
        self.base_model.eval()
        
        self.adapters = {}
        self.heads = {}
        self.layers = {}
        
        base_config = self.base_model.config
        hidden_size = base_config.hidden_size
        vocab_size = base_config.vocab_size
        
        self.adapters[first_adapter_name] = temp_cats.adapter_model
        self.heads[first_adapter_name] = temp_cats.head_model
        self.layers[first_adapter_name] = first_adapter_config['layer']
        
        del temp_cats
        
        step_num = 2
        for adapter_name, adapter_config in adapter_configs.items():
            if adapter_name == first_adapter_name:
                continue
            
            adapter_path = adapter_config['path']
            adapter_layer = adapter_config['layer']
            
            temp_cats = CATSModel(
                base_model_path,
                adapter_path,
                args,
                EARLY_STOP_LAYER=self.max_early_stop_layer
            )
            
            adapter_model = temp_cats.adapter_model
            head_model = temp_cats.head_model
            
            actual_num_layers = len(adapter_model.layers)
            if actual_num_layers != 1:
                raise ValueError(
                    f"adapter '{adapter_name}' has {actual_num_layers} layers, expected 1."
                )
            
            del temp_cats
            
            self.adapters[adapter_name] = adapter_model
            self.heads[adapter_name] = head_model
            self.layers[adapter_name] = adapter_layer
            
            step_num += 1
    
    def get_adapter(self, adapter_name: str):
        if adapter_name not in self.adapters:
            raise ValueError(f"Adapter '{adapter_name}' not found. Available: {list(self.adapters.keys())}")
        return self.adapters[adapter_name]
    
    def get_head(self, adapter_name: str):
        if adapter_name not in self.heads:
            raise ValueError(f"Head for '{adapter_name}' not found. Available: {list(self.heads.keys())}")
        return self.heads[adapter_name]
    
    def get_layer(self, adapter_name: str):
        if adapter_name not in self.layers:
            raise ValueError(f"Layer for '{adapter_name}' not found. Available: {list(self.layers.keys())}")
        return self.layers[adapter_name]
    
    @property
    def training(self):
        return self.base_model.training
    
    @property
    def device(self):
        return self.base_model.device
    
    def eval(self):
        self.base_model.eval()
        for adapter in self.adapters.values():
            adapter.eval()
        for head in self.heads.values():
            if hasattr(head, 'eval'):
                head.eval()
        return self

def cats_forward_hybrid_evaluation(
    inputs,
    model,
    tokenizer,
    max_new_tokens,
    do_sample=False,
    max_length=2048,
    draft_layer=3,
    shallow_layer=10,
    SPECULATIVE_DECODING_STEPS=6,
    threshold=0.6,
    sv_passes=1,
    sv_block_size=16,
    min_target_verify=2,
    sv_maintain_kv=True,
    disable_sv=False,
    tree_topk=5,
    total_tokens=-1,
    sv_score_bonus=2.0,
    **kwargs
):
    global _global_metrics

    # Propagate typical-acceptance kwargs to module-level _config (Ray subprocesses).
    for k in ("typical_epsilon", "typical_tau", "typical_alpha", "temperature"):
        if k in kwargs:
            _config[k] = kwargs[k]

    draft_adapter = model.get_adapter('draft')
    shallow_adapter = model.get_adapter('shallow')
    draft_head = model.get_head('draft')
    shallow_head = model.get_head('shallow')
    
    DRAFT_EXIT_LAYER = model.get_layer('draft')
    SHALLOW_EXIT_LAYER = model.get_layer('shallow')
    
    context_tokens = inputs.input_ids
    device = context_tokens.device
    token_eos = tokenizer.eos_token_id
    
    batch_size, context_length = context_tokens.shape
    
    hidden_size = model.base_model.config.hidden_size
    hidden_dtype = model.base_model.model.layers[0].self_attn.q_proj.weight.dtype

    max_position_embeddings = model.base_model.config.max_position_embeddings
    actual_max_length = min(max_length, max_position_embeddings)
    max_length = actual_max_length
    
    global_tokens = torch.ones((batch_size, max_length), dtype=torch.long, device=device) * token_eos
    global_position_ids = torch.LongTensor([[i for i in range(max_length)]]).to(device)
    
    accept_length_list = [1]
    start_index = context_length
    global_tokens[:, :start_index] = context_tokens

    # ============ Initialization ============
    with torch.no_grad():
        position_ids = global_position_ids[:, :start_index]
        
        output = model.base_model(
            context_tokens, 
            position_ids=position_ids, 
            output_hidden_states=True
        )
        
        model.base_model.past_key_values = list(output.past_key_values)
        
        logits = output.logits
        global_tokens[:, start_index] = torch.argmax(logits[:, -1, :], dim=-1).item()
        
        hidden_state_draft = output.hidden_states[DRAFT_EXIT_LAYER]
        _, draft_adapter_kv = draft_adapter.forward_early_stop(
            inputs_embeds=hidden_state_draft,
            position_ids=None,
            use_cache=True
        )
        
        hidden_state_shallow = output.hidden_states[SHALLOW_EXIT_LAYER]
        _, shallow_adapter_kv = shallow_adapter.forward_early_stop(
            inputs_embeds=hidden_state_shallow,
            position_ids=position_ids,
            use_cache=True
        )

    total_inference_steps = 0

    loop_records = []

    # ============ Main loop ============
    with torch.no_grad():
        max_infer_steps = min(max_length, start_index + max_new_tokens)
        stop = False
        
        exited_hidden_states_draft = None

        try:
            TOTAL_DRAFT_STEPS = SPECULATIVE_DECODING_STEPS * max(1, sv_passes)
            SV_BLOCK_SIZE = sv_block_size
            K_tree = max(1, tree_topk)
            
            pending_sv_tokens = []
            pending_shallow_mask = []
            pending_sv_hidden = torch.zeros(
                (1, 0, hidden_size),
                dtype=hidden_dtype,
                device=device
            )
            tree_structure = None
            tree_accepted_tokens = []
            tree_accepted_hidden = None
            _sv_case = None
            
            loop_threshold = max_infer_steps - 1 - TOTAL_DRAFT_STEPS
            
            max_position = model.base_model.config.max_position_embeddings
            loop_threshold = min(loop_threshold, max_position - TOTAL_DRAFT_STEPS)
            
            while start_index < loop_threshold:
                _truncate_base_past_kv(model.base_model, start_index, start_layer=0, end_layer=None)

                draft_adapter_kv = _truncate_kv_cache(draft_adapter_kv, start_index)
                shallow_adapter_kv = _truncate_kv_cache(shallow_adapter_kv, start_index)
                # Note: do NOT reset exited_hidden_states_draft here;
                # it may carry catch-up features from the previous iteration's tree verification.
                # The draft loop's step==0 handles the fresh reset internally.
                
                step_num = total_inference_steps + 1
                
                start_index_copy = start_index
                end_index = start_index + 1
                shallow_accept_mask = []
                
                # ============ STEP 1: Draft Generation ============
                
                inner_loop_count = 0
                MAX_INNER_LOOPS = 20
                
                while len(pending_sv_tokens) < 1:
                    inner_loop_count += 1
                    round_context_start = start_index

                    shallow_kv_len_before = _get_seq_len_from_kv(shallow_adapter_kv)
                    
                    if shallow_kv_len_before < round_context_start:
                        missing_start = shallow_kv_len_before
                        missing_end = round_context_start
                        missing_tokens = global_tokens[:, missing_start:missing_end]
                        missing_len = missing_end - missing_start
                        missing_position_ids = global_position_ids[:, missing_start:missing_end]
                        
                        original_early_exit_layer = model.base_model.early_exit_layer
                        model.base_model.early_exit_layer = SHALLOW_EXIT_LAYER
                        
                        missing_hidden_sv = model.base_model.forward_draft_or_large_model(
                            in_tokens_small=missing_tokens,
                            position_ids=missing_position_ids
                        )
                        
                        model.base_model.early_exit_layer = original_early_exit_layer
                        _truncate_base_past_kv(model.base_model, round_context_start,
                                               start_layer=0, end_layer=SHALLOW_EXIT_LAYER)
                        shallow_out = shallow_adapter.forward_early_stop(
                            inputs_embeds=missing_hidden_sv,
                            position_ids=missing_position_ids,
                            past_key_values=shallow_adapter_kv,
                            use_cache=True
                        )
                        shallow_adapter_kv = shallow_out[1] if isinstance(shallow_out, tuple) and len(shallow_out) == 2 else shallow_out.past_key_values
                        
                    elif shallow_kv_len_before > round_context_start:
                        shallow_adapter_kv = _truncate_kv_cache(shallow_adapter_kv, round_context_start)
                    
                    shallow_kv_len_final = _get_seq_len_from_kv(shallow_adapter_kv)
                    assert shallow_kv_len_final == round_context_start, \
                        f"SV adapter KV cache length ({shallow_kv_len_final}) must equal round_context_start ({round_context_start})"
                    
                    draft_tokens_list = []
                    dm_topk_tokens_list = []
                    dm_topk_logprobs_list = []
                    predict_score = 1.0
                    
                    for step in range(TOTAL_DRAFT_STEPS):
                        draft_kv_len = draft_adapter_kv[0][0].shape[2]
                        in_tokens = global_tokens[:, end_index-1:end_index]

                        if draft_kv_len < end_index - 1 and exited_hidden_states_draft is not None:
                            position_ids = global_position_ids[:, start_index-1:end_index]
                            hidden_state_draft_last = exited_hidden_states_draft[:, -1:, :]
                        else:
                            position_ids = global_position_ids[:, end_index-1:end_index]
                            hidden_state_draft_last = None
                        
                        original_early_exit_layer = model.base_model.early_exit_layer
                        model.base_model.early_exit_layer = DRAFT_EXIT_LAYER

                        hidden_state_draft = model.base_model.forward_draft_or_large_model(
                            in_tokens_small=in_tokens[:, -1:],
                            position_ids=position_ids[:, -1:]
                        )

                        model.base_model.early_exit_layer = original_early_exit_layer
                        
                        if step == 0:
                            exited_hidden_states_draft = None
                        
                        exited_hidden_states_draft = hidden_state_draft if exited_hidden_states_draft is None else \
                            torch.cat([exited_hidden_states_draft, hidden_state_draft], dim=1)
                        
                        if hidden_state_draft_last is not None:
                            hidden_state_draft_input = torch.cat([hidden_state_draft_last, hidden_state_draft], dim=1)
                        else:
                            hidden_state_draft_input = hidden_state_draft
                        
                        hidden_state_draft_out, draft_adapter_kv = draft_adapter.forward_early_stop(
                            inputs_embeds=hidden_state_draft_input,
                            position_ids=None,
                            past_key_values=draft_adapter_kv,
                            use_cache=True
                        )

                        predict_logits = draft_head(hidden_state_draft_out[:, -1:, :]).float()
                        log_probs = torch.log_softmax(predict_logits[:, -1, :], dim=-1)
                        topk_lp, topk_idx = torch.topk(log_probs, k=K_tree, dim=-1)
                        predicted_token = topk_idx[0, 0]
                        predict_score = topk_lp[0, 0].exp().item()
                        
                        global_tokens[:, end_index] = predicted_token
                        draft_tokens_list.append(predicted_token.item())
                        dm_topk_tokens_list.append(topk_idx[0])
                        dm_topk_logprobs_list.append(topk_lp[0])
                        
                        end_index += 1
                        
                        if step == TOTAL_DRAFT_STEPS - 1:
                            break
                        
                        if step > 0 and predict_score < threshold:
                            break
                
                    num_draft_tokens = len(draft_tokens_list)
                    draft_hidden_seq_len = exited_hidden_states_draft.shape[1] if exited_hidden_states_draft is not None else 0
                    
                    # batch
                    if draft_hidden_seq_len == 0:
                        shallow_hidden_batch = exited_hidden_states_draft[:, :0, :]
                        position_ids_shallow = global_position_ids[:, start_index:start_index]
                        _seq_len = 0
                    else:
                        shallow_hidden_batch = exited_hidden_states_draft
                        position_ids_shallow = global_position_ids[:, start_index:start_index+draft_hidden_seq_len]
                        _seq_len = draft_hidden_seq_len
                    
                    _bs = shallow_hidden_batch.shape[0]

                    if _seq_len > 0:
                        if DRAFT_EXIT_LAYER < len(model.base_model.past_key_values) and \
                           model.base_model.past_key_values[DRAFT_EXIT_LAYER] is not None and \
                           model.base_model.past_key_values[DRAFT_EXIT_LAYER][0] is not None:
                            _past_len = model.base_model.past_key_values[DRAFT_EXIT_LAYER][0].shape[2]
                        else:
                            _past_len = start_index
                        
                        _mask_full = torch.ones((_bs, _past_len + _seq_len), dtype=torch.bool, device=device)
                        attention_mask = model.base_model.model._prepare_decoder_attention_mask(
                            _mask_full, (_bs, _seq_len), shallow_hidden_batch, _past_len
                        )
                    else:
                        attention_mask = None
                    
                    # Forward layers DRAFT_EXIT_LAYER to SHALLOW_EXIT_LAYER-1 (draft → sv)
                    for layer_idx in range(DRAFT_EXIT_LAYER, SHALLOW_EXIT_LAYER):
                        layer = model.base_model.model.layers[layer_idx]
                        layer_outputs = layer(
                            shallow_hidden_batch,
                            attention_mask=attention_mask,
                            position_ids=position_ids_shallow,
                            past_key_value=model.base_model.past_key_values[layer_idx],
                            use_cache=True
                        )
                        
                        shallow_hidden_batch = layer_outputs[0]
                        model.base_model.past_key_values[layer_idx] = layer_outputs[1]

                    base_hidden_sv = shallow_hidden_batch.clone()
                    sv_verification_hidden = shallow_hidden_batch
                    
                    shallow_kv_len = _get_seq_len_from_kv(shallow_adapter_kv)
                    
                    assert shallow_kv_len == round_context_start, \
                        f"SV adapter KV cache length ({shallow_kv_len}) must equal round_context_start ({round_context_start})"
                    
                    position_ids_sv = global_position_ids[:, round_context_start:round_context_start + draft_hidden_seq_len]
                    
                    shallow_out = shallow_adapter.forward_early_stop(
                        inputs_embeds=sv_verification_hidden,
                        position_ids=position_ids_sv,
                        past_key_values=shallow_adapter_kv,
                        use_cache=True
                    )
                    
                    shallow_adapter_kv = shallow_out[1] if isinstance(shallow_out, tuple) and len(shallow_out) == 2 else shallow_out.past_key_values
                    
                    sv_output_hidden = shallow_out[0] if isinstance(shallow_out, tuple) else shallow_out
                    
                    shallow_logits = shallow_head(sv_output_hidden).float()
                    shallow_pred_tokens = torch.argmax(shallow_logits, dim=-1)

                    output_length = draft_hidden_seq_len
                    mismatch_list: List[Tuple[int, int]] = []

                    for i in range(output_length):
                        sv_pred_token = shallow_pred_tokens[0, i].item()
                        draft_token_at_pos = global_tokens[0, round_context_start + 1 + i].item()

                        if i < len(draft_tokens_list) and sv_pred_token != draft_token_at_pos:
                            mismatch_list.append((i, sv_pred_token))

                    has_mismatch = len(mismatch_list) > 0
                    mismatch_position = mismatch_list[0][0] if has_mismatch else -1
                    sv_corrected_token = mismatch_list[0][1] if has_mismatch else None
                    
                    # ========== Case 1: SV accepts all draft tokens ==========
                    if not has_mismatch:
                        tree_structure = None
                        tree_accepted_hidden = None

                        last_token_idx = round_context_start + 1 + (output_length - 1)
                        last_token = global_tokens[0, last_token_idx].item()
                        
                        last_token_tensor = global_tokens[:, last_token_idx:last_token_idx+1]
                        last_token_position = global_position_ids[:, last_token_idx:last_token_idx+1]

                        original_early_exit_layer = model.base_model.early_exit_layer
                        model.base_model.early_exit_layer = DRAFT_EXIT_LAYER
                        last_token_hidden_layer3 = model.base_model.forward_draft_or_large_model(
                            in_tokens_small=last_token_tensor,
                            position_ids=last_token_position
                        )
                        model.base_model.early_exit_layer = original_early_exit_layer

                        draft_adapter_input = last_token_hidden_layer3  # only h(d_last)

                        draft_adapter_out, draft_adapter_kv = draft_adapter.forward_early_stop(
                            inputs_embeds=draft_adapter_input,
                            position_ids=None,
                            past_key_values=draft_adapter_kv,
                            use_cache=True
                        )

                        predict_logits_bonus = draft_head(draft_adapter_out[:, -1:, :]).float()
                        bonus_lp = torch.log_softmax(predict_logits_bonus[:, -1, :], dim=-1)
                        bonus_topk_lp, bonus_topk_idx = torch.topk(bonus_lp, k=K_tree, dim=-1)
                        bonus_token = bonus_topk_idx[0, 0].item()
                        bonus_topk_tokens = bonus_topk_idx[0]
                        bonus_topk_logprobs = bonus_topk_lp[0]

                        bonus_token_idx = last_token_idx + 1
                        global_tokens[0, bonus_token_idx] = bonus_token

                        sv_accept_count = output_length + 1
                        # Store info for dynamic tree building later
                        _sv_case = 'all_pass'
                        _sv_mismatch_list = []
                        _sv_corr_topk_tokens = []
                        _sv_corr_topk_logprobs = []
                        _sv_bonus_token = bonus_token
                        _sv_bonus_topk_tokens = bonus_topk_tokens
                        _sv_bonus_topk_logprobs = bonus_topk_logprobs
                        
                    # ========== Case 2: SV mismatch (multi-branch) ==========
                    elif has_mismatch:
                        actual_accepted_count = mismatch_position
                        num_corrections = len(mismatch_list)
                        mismatch_positions = [pos for pos, _ in mismatch_list]
                        corrected_tokens = [ct for _, ct in mismatch_list]

                        tree_structure = None
                        tree_accepted_hidden = None

                        # Phase A: write ALL corrected tokens to global_tokens
                        corrected_token_indices = []
                        for pos, ct in mismatch_list:
                            ct_idx = round_context_start + 1 + pos
                            global_tokens[0, ct_idx] = ct
                            corrected_token_indices.append(ct_idx)

                        # Phase B: batch forward [draft_last, c_k1, ..., c_kN] through layers 0..DRAFT_EXIT_LAYER-1
                        draft_last_token_idx = round_context_start + 1 + (len(draft_tokens_list) - 1)
                        draft_last_token = draft_tokens_list[-1]

                        if model.base_model.past_key_values[0] is not None and \
                           model.base_model.past_key_values[0][0] is not None:
                            _past_len_draft = model.base_model.past_key_values[0][0].shape[2]
                        else:
                            _past_len_draft = round_context_start + 1

                        batch_token_list = [draft_last_token] + corrected_tokens
                        batch_tokens_draft = torch.tensor(
                            [batch_token_list], device=device, dtype=torch.long
                        )
                        batch_pos_list = [global_position_ids[:, draft_last_token_idx:draft_last_token_idx+1]]
                        for ct_idx in corrected_token_indices:
                            batch_pos_list.append(global_position_ids[:, ct_idx:ct_idx+1])
                        batch_pos_ids_draft = torch.cat(batch_pos_list, dim=1)

                        _main_branch_len_draft = len(draft_tokens_list)

                        batch_hidden_draft_out = _forward_tokens_tree_attention_to_draft(
                            model,
                            batch_tokens_draft,
                            batch_pos_ids_draft,
                            device,
                            DRAFT_EXIT_LAYER,
                            round_context_start,
                            _past_len_draft,
                            _main_branch_len_draft,
                            mismatch_positions,
                        )

                        draft_last_hidden = batch_hidden_draft_out[:, 0:1, :]
                        correction_hiddens_draft = [
                            batch_hidden_draft_out[:, 1+i:2+i, :] for i in range(num_corrections)
                        ]

                        # Phase C: draft adapter generates bonus token with top-K
                        draft_adapter_input = draft_last_hidden  # only h(d_last)

                        hidden_state_draft_out, draft_adapter_kv = draft_adapter.forward_early_stop(
                            inputs_embeds=draft_adapter_input,
                            position_ids=None,
                            past_key_values=draft_adapter_kv,
                            use_cache=True
                        )

                        predict_logits_bonus = draft_head(hidden_state_draft_out[:, -1:, :]).float()
                        bonus_lp = torch.log_softmax(predict_logits_bonus[:, -1, :], dim=-1)
                        bonus_topk_lp, bonus_topk_idx = torch.topk(bonus_lp, k=K_tree, dim=-1)
                        draft_bonus_token = bonus_topk_idx[0, 0].item()
                        bonus_topk_tokens = bonus_topk_idx[0]
                        bonus_topk_logprobs = bonus_topk_lp[0]

                        draft_bonus_token_idx = draft_last_token_idx + 1
                        global_tokens[0, draft_bonus_token_idx] = draft_bonus_token

                        # Phase C2: correction continuation drafting
                        # For each correction, run adapter forward to get continuation token + top-K
                        corr_cont_topk_tokens = []
                        corr_cont_topk_logprobs = []
                        for ci in range(num_corrections):
                            corr_hidden = correction_hiddens_draft[ci]  # [1, 1, hidden]
                            corr_adapter_out = draft_adapter.forward_early_stop(
                                inputs_embeds=corr_hidden,
                                position_ids=None,
                                past_key_values=_truncate_kv_cache(draft_adapter_kv,
                                    round_context_start + 1 + mismatch_positions[ci]),
                                use_cache=False
                            )
                            corr_logits = draft_head(corr_adapter_out[:, -1:, :]).float()
                            corr_lp = torch.log_softmax(corr_logits[:, -1, :], dim=-1)
                            corr_topk_lp, corr_topk_idx = torch.topk(corr_lp, k=K_tree, dim=-1)
                            corr_cont_topk_tokens.append(corr_topk_idx[0])
                            corr_cont_topk_logprobs.append(corr_topk_lp[0])

                        sv_accept_count = actual_accepted_count
                        _sv_case = 'mismatch'
                        _sv_mismatch_list = mismatch_list
                        _sv_corr_topk_tokens = corr_cont_topk_tokens
                        _sv_corr_topk_logprobs = corr_cont_topk_logprobs
                        _sv_bonus_token = draft_bonus_token
                        _sv_bonus_topk_tokens = bonus_topk_tokens
                        _sv_bonus_topk_logprobs = bonus_topk_logprobs
                        
                        
                    # ========== Common bookkeeping after SV verification ==========
                    for i in range(sv_accept_count):
                        accepted_token = global_tokens[0, round_context_start + 1 + i].item()
                        pending_sv_tokens.append(accepted_token)
                        pending_shallow_mask.append(1)

                    if _sv_case == 'mismatch' and sv_accept_count == 0:
                        break

                # ========== Step 2: Target Model Verification (EAGLE-style dynamic tree) ==========
                
                if len(pending_sv_tokens) == 0 and _sv_case not in ('mismatch', 'all_pass'):
                    continue
                
                tm_start_index = start_index
                num_draft_tokens = len(pending_sv_tokens)
                shallow_accept_mask = pending_shallow_mask[:]

                _typical_tau = _config.get('typical_tau', 0.3)
                _typical_alpha = _config.get('typical_alpha', 0.09)
                _typical_eps = _config.get('typical_epsilon', 0.0)
                _temperature = _config.get('temperature', 0.7)

                tree_tokens_t, medusa_mask_t, pos_offsets, retrieve_indices, D_trunk = \
                        build_dynamic_tree_v2(
                            bonus_token=global_tokens[0, tm_start_index].item(),
                            draft_tokens_list=draft_tokens_list,
                            dm_topk_tokens=dm_topk_tokens_list,
                            dm_topk_logprobs=dm_topk_logprobs_list,
                            sv_corrections=_sv_mismatch_list,
                            corr_topk_tokens=_sv_corr_topk_tokens,
                            corr_topk_logprobs=_sv_corr_topk_logprobs,
                            total_tokens=total_tokens,
                            device=device,
                            sv_score_bonus=sv_score_bonus,
                        )

                T_tree = tree_tokens_t.shape[1]
                effective_draft_tokens = tree_tokens_t[0].tolist()

                _truncate_base_past_kv(model.base_model, tm_start_index, start_layer=0, end_layer=None)

                original_early_exit_layer = model.base_model.early_exit_layer
                total_layers = len(model.base_model.model.layers)

                model.base_model.model.medusa_mask = medusa_mask_t
                tree_position_ids = (pos_offsets.unsqueeze(0) + tm_start_index).to(device)

                model.base_model.early_exit_layer = DRAFT_EXIT_LAYER
                exited_tree_hidden = model.base_model.forward_draft_or_large_model(
                    in_tokens_small=tree_tokens_t, position_ids=tree_position_ids
                )
                _, tree_hidden_normed = model.base_model.forward_draft_or_large_model(
                    in_features_large=exited_tree_hidden, position_ids=tree_position_ids
                )

                model.base_model.early_exit_layer = original_early_exit_layer
                model.base_model.model.medusa_mask = None

                target_logits = model.base_model.lm_head(tree_hidden_normed).float()

                _posterior_threshold = _typical_tau if _typical_eps <= 0.0 else _typical_eps
                _posterior_alpha = _typical_alpha

                best_path, accept_length = evaluate_posterior_tree(
                    target_logits,
                    tree_tokens_t,
                    retrieve_indices,
                    _posterior_threshold,
                    _posterior_alpha,
                    _temperature,
                )
                select_tree_indices = retrieve_indices[best_path, : accept_length + 1]

                eos_cutoff = None
                for i in range(1, accept_length + 1):
                    tok = int(tree_tokens_t[0, select_tree_indices[i]].item())
                    if tok == token_eos:
                        eos_cutoff = i
                        break

                if eos_cutoff is not None:
                    for i in range(1, eos_cutoff + 1):
                        global_tokens[0, tm_start_index + i] = tree_tokens_t[0, select_tree_indices[i]]
                    accept_length = eos_cutoff
                    new_start = tm_start_index + accept_length
                    stop = True
                else:
                    for i in range(1, accept_length + 1):
                        global_tokens[0, tm_start_index + i] = tree_tokens_t[0, select_tree_indices[i]]
                    last_idx_on_path = int(select_tree_indices[accept_length].item())
                    next_bonus = int(target_logits[0, last_idx_on_path, :].argmax().item())
                    global_tokens[0, tm_start_index + accept_length + 1] = next_bonus
                    new_start = tm_start_index + accept_length + 1
                    if next_bonus == token_eos:
                        stop = True

                accepted_by_target = new_start - tm_start_index

                keep_from_tree = new_start - tm_start_index
                select_kv_tree = select_tree_indices[:keep_from_tree].clamp(min=0).to(device)
                past_key_values_new = []
                for _k, _v in model.base_model.past_key_values:
                    prompt_k = _k[:, :, :tm_start_index, :]
                    prompt_v = _v[:, :, :tm_start_index, :]
                    sel_k = _k[:, :, tm_start_index + select_kv_tree, :]
                    sel_v = _v[:, :, tm_start_index + select_kv_tree, :]
                    past_key_values_new.append(
                        (torch.cat([prompt_k, sel_k], dim=2),
                         torch.cat([prompt_v, sel_v], dim=2))
                    )
                model.base_model.past_key_values = past_key_values_new

                start_index = new_start

                last_accepted_tree_idx = (
                    int(select_tree_indices[accept_length].item()) if accept_length >= 1 else 0
                )
                if last_accepted_tree_idx >= D_trunk:
                    adapter_target_len = start_index - 1
                    catchup_feature = exited_tree_hidden[:, last_accepted_tree_idx:last_accepted_tree_idx + 1, :]
                else:
                    adapter_target_len = start_index
                    catchup_feature = None

                _first_branch_pos = None
                _intermediate_branch_features = []
                for _i in range(1, accept_length):
                    _tree_idx = int(select_tree_indices[_i].item())
                    if _tree_idx > D_trunk:
                        if _first_branch_pos is None:
                            _first_branch_pos = _i
                        _intermediate_branch_features.append(
                            exited_tree_hidden[:, _tree_idx:_tree_idx + 1, :])

                if _first_branch_pos is not None and _intermediate_branch_features:
                    _fix_truncate = tm_start_index + _first_branch_pos
                    draft_adapter_kv = _truncate_kv_cache(draft_adapter_kv, _fix_truncate)
                    _branch_hidden = torch.cat(_intermediate_branch_features, dim=1)
                    _, draft_adapter_kv = draft_adapter.forward_early_stop(
                        inputs_embeds=_branch_hidden,
                        position_ids=None,
                        past_key_values=draft_adapter_kv,
                        use_cache=True
                    )

                draft_adapter_kv = _truncate_kv_cache(draft_adapter_kv, adapter_target_len)
                shallow_adapter_kv = _truncate_kv_cache(shallow_adapter_kv, start_index)

                if catchup_feature is not None:
                    exited_hidden_states_draft = catchup_feature

                # ========== Bookkeeping ==========
                metrics = EvaluationMetrics()
                sv_accept_prefix = len([m for m in shallow_accept_mask if m == 1])
                if _global_metrics is not None:
                    _global_metrics.add_step(metrics, num_draft_tokens, accepted_by_target, sv_accept_prefix)
                
                sv_log_entry = {
                    "step": int(step_num),
                    "loop_index": int(total_inference_steps),
                    "draft_tokens": effective_draft_tokens[:num_draft_tokens] if isinstance(effective_draft_tokens, list) else list(effective_draft_tokens)[:num_draft_tokens],
                    "sv_draft_length": int(len(pending_sv_tokens)),
                    "accepted_by_target": int(accepted_by_target),
                    "start_index": int(start_index)
                }
                _sv_log_records.append(sv_log_entry)
                
                accept_length_list.append(accepted_by_target)
                
                pending_sv_tokens = []
                pending_shallow_mask = []
                pending_sv_hidden = torch.zeros((1, 0, hidden_size), dtype=hidden_dtype, device=device)

                total_inference_steps += 1

                if stop:
                    break
                
                if start_index >= loop_threshold:
                    break
        
        except Exception:
            import traceback
            traceback.print_exc()
            raise
    
    output_ids = global_tokens[0, :start_index + 1].tolist()
    new_token = start_index - context_length + 1
    idx = len(accept_length_list) - 1

    return [output_ids], new_token, idx, accept_length_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--draft-adapter-path", type=str, required=True)
    parser.add_argument("--shallow-adapter-path", type=str, required=True)
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--bench-name", type=str, default="mt_bench")
    parser.add_argument("--question-begin", type=int, default=None)
    parser.add_argument("--question-end", type=int, default=None)
    parser.add_argument("--answer-file", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--num-choices", type=int, default=1)
    parser.add_argument("--num-gpus-per-model", type=int, default=1)
    parser.add_argument("--num-gpus-total", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--draft-layer", type=int, default=3)
    parser.add_argument("--shallow-layer", type=int, default=10)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--sv-passes", type=int, default=1)
    parser.add_argument("--sv-block-size", type=int, default=16)
    parser.add_argument("--min-target-verify", type=int, default=2)
    parser.add_argument("--sv-log-file", type=str, default=None)
    parser.add_argument(
        "--sv-maintain-kv",
        dest="sv_maintain_kv",
        nargs="?",
        const=True,
        default=True,
        type=lambda s: str(s).lower() in ("true", "1", "yes", "y"),
    )
    parser.add_argument(
        "--no-sv-maintain-kv",
        dest="sv_maintain_kv",
        action="store_false",
    )
    parser.add_argument("--disable-sv", action="store_true")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float32", "float64", "float16", "bfloat16"])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--num-runs", type=int, default=3, dest="num_runs",
                        help="Number of evaluation runs (default 3, i.e. run 0, 1, 2).")
    parser.add_argument(
        "--typical-epsilon",
        type=float,
        default=0.0,
        dest="typical_epsilon",
        help="Legacy: when set, used as typical_tau (Medusa-aligned). 0 = disabled.",
    )
    parser.add_argument(
        "--typical-tau",
        type=float,
        default=0.3,
        dest="typical_tau",
        help="Typical Acceptance tau (Medusa posterior_threshold). threshold = min(tau, alpha*exp(-H)).",
    )
    parser.add_argument(
        "--typical-alpha",
        type=float,
        default=0.09,
        dest="typical_alpha",
        help="Typical Acceptance alpha (Medusa posterior_alpha). threshold = min(tau, alpha*exp(-H)).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        dest="temperature",
        help="Temperature for typical acceptance: P = softmax(logits/temperature). 1.0 = no scaling; 0.7 = Medusa CLI default.",
    )
    parser.add_argument("--tree-topk", type=int, default=5, dest="tree_topk",
                        help="Top-K candidates per draft step for dynamic tree construction.")
    parser.add_argument("--total-tokens", type=int, default=-1, dest="total_tokens",
                        help="Total tree node budget for dynamic tree (-1 = auto = D*K+1).")
    parser.add_argument("--sv-score-bonus", type=float, default=2.0, dest="sv_score_bonus",
                        help="Score bonus for SV correction tokens in dynamic tree.")

    args = parser.parse_args()

    question_file = f"data/question_mtbench.jsonl" # MT-bench, you can change to another test bench.
    
    model = MultiAdapterCATSModel(
        base_model_path=args.model_path,
        adapter_configs={
            'draft': {
                'path': args.draft_adapter_path,
                'layer': args.draft_layer
            },
            'shallow': {
                'path': args.shallow_adapter_path,
                'layer': args.shallow_layer
            }
        },
        args=args
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    output_dir = f"data/{args.bench_name}/{args.model_id}"
    os.makedirs(output_dir, exist_ok=True)

    _config['output_dir'] = output_dir
    _config['model_id'] = args.model_id
    _config['bench_name'] = args.bench_name
    _config['typical_epsilon'] = max(0.0, args.typical_epsilon)
    _config['typical_tau'] = max(0.0, args.typical_tau)
    _config['typical_alpha'] = max(0.0, args.typical_alpha)
    _config['temperature'] = max(1e-8, args.temperature)

    all_runs_metrics = []
    runs_details = []
    num_runs = max(1, args.num_runs)

    for run in range(num_runs):
        answer_file = f"{output_dir}/{run}.jsonl"
        print(f"Run {run + 1}/{num_runs}: Output to {answer_file}")
        _sv_log_records.clear()
        
        _global_metrics = GlobalMetrics()

        run_eval(
            model=model,
            tokenizer=tokenizer,
            forward_func=cats_forward_hybrid_evaluation,
            model_id=args.model_id,
            question_file=question_file,
            question_begin=args.question_begin,
            question_end=args.question_end,
            answer_file=answer_file,
            max_new_tokens=args.max_new_tokens,
            num_choices=args.num_choices,
            num_gpus_per_model=args.num_gpus_per_model,
            num_gpus_total=args.num_gpus_total,
            do_sample=False,
            threshold=args.threshold,
            SPECULATIVE_DECODING_STEPS=args.steps,
            draft_layer=args.draft_layer,
            shallow_layer=args.shallow_layer,
            sv_passes=args.sv_passes,
            sv_block_size=args.sv_block_size,
            min_target_verify=args.min_target_verify,
            sv_maintain_kv=args.sv_maintain_kv,
            disable_sv=args.disable_sv,
            typical_epsilon=args.typical_epsilon,
            typical_tau=args.typical_tau,
            typical_alpha=args.typical_alpha,
            temperature=args.temperature,
            tree_topk=args.tree_topk,
            total_tokens=args.total_tokens,
            sv_score_bonus=args.sv_score_bonus,
        )

        reorg_answer_file(answer_file)

        all_runs_metrics.append(_global_metrics)
        run_steps = len(_global_metrics.accepted_tokens_per_step)
        run_mean_accepted = (sum(_global_metrics.accepted_tokens_per_step) / run_steps) if run_steps > 0 else 0.0
        print("#Mean accepted tokens: ", run_mean_accepted)
        run_mean_sv = (sum(_global_metrics.sv_accepted_tokens_per_step) / run_steps) if run_steps > 0 else 0.0
        run_acc, run_prec, run_recall, run_f1 = _global_metrics.total_metrics.compute()

        runs_details.append({
            'run_index': run + 1,
            'steps': run_steps,
            'mean_accepted_tokens_per_step': run_mean_accepted,
            'mean_sv_accepted_tokens_per_step': run_mean_sv,
            'shallow_verifier_stats': {
                'tp': _global_metrics.total_metrics.tp,
                'tn': _global_metrics.total_metrics.tn,
                'fp': _global_metrics.total_metrics.fp,
                'fn': _global_metrics.total_metrics.fn,
                'accuracy': run_acc,
                'precision': run_prec,
                'recall': run_recall,
                'f1_score': run_f1
            }
        })
    
    combined_metrics = GlobalMetrics()
    for run_metrics in all_runs_metrics:
        combined_metrics.total_metrics.tp += run_metrics.total_metrics.tp
        combined_metrics.total_metrics.tn += run_metrics.total_metrics.tn
        combined_metrics.total_metrics.fp += run_metrics.total_metrics.fp
        combined_metrics.total_metrics.fn += run_metrics.total_metrics.fn
        combined_metrics.draft_tokens_per_step.extend(run_metrics.draft_tokens_per_step)
        combined_metrics.accepted_tokens_per_step.extend(run_metrics.accepted_tokens_per_step)
        combined_metrics.sv_accepted_tokens_per_step.extend(run_metrics.sv_accepted_tokens_per_step)
    
    combined_metrics.print_summary(args.steps)
    
    metrics_file = f"{output_dir}/verification_metrics.json"
    total_tokens = (combined_metrics.total_metrics.tp + 
                   combined_metrics.total_metrics.tn +
                   combined_metrics.total_metrics.fp +
                   combined_metrics.total_metrics.fn)
    
    acc, prec, recall, f1 = combined_metrics.total_metrics.compute()
    
    metrics_summary = {
        'total_runs': num_runs,
        'speculative_decoding_steps': args.steps,
        'draft_layer': args.draft_layer,
        'shallow_layer': args.shallow_layer,
        'threshold': args.threshold,
        'sv_passes': args.sv_passes,
        'total_tokens_verified': total_tokens,
        'runs': runs_details,
        'shallow_verifier_stats': {
            'tp': combined_metrics.total_metrics.tp,
            'tn': combined_metrics.total_metrics.tn,
            'fp': combined_metrics.total_metrics.fp,
            'fn': combined_metrics.total_metrics.fn,
            'accuracy': acc,
            'precision': prec,
            'recall': recall,
            'f1_score': f1
        },
        'draft_generation_stats': {
            'total_steps': len(combined_metrics.draft_tokens_per_step),
            'mean_draft_tokens_per_step': sum(combined_metrics.draft_tokens_per_step) / len(combined_metrics.draft_tokens_per_step) if combined_metrics.draft_tokens_per_step else 0,
            'mean_accepted_tokens_per_step': sum(combined_metrics.accepted_tokens_per_step) / len(combined_metrics.accepted_tokens_per_step) if combined_metrics.accepted_tokens_per_step else 0,
        }
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics_summary, f, indent=2)