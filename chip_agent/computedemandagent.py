"""Computing Demand Agent: generates per-operator computing demand CSVs for LLM models.

Uses an OpenAI LLM agent to read model configs and compute per-operator FLOP counts,
tensor shapes, and execution counts. The LLM applies architectural knowledge and
sharding formulas to derive values from first principles.
"""

import argparse
import json
import math
import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent / ".env")

_NEUSIM_RESULTS = Path(__file__).resolve().parent.parent / "NeuSim" / "results" / "raw"

OUTPUT_COLUMNS = [
    "Description",
    "Count",
    "FLOP Count",
    "Input Tensor Shapes",
    "Output Tensor Shapes",
]

TOOLS = [{
    "type": "function",
    "function": {
        "name": "compute_demand",
        "description": (
            "Compute precise Count, FLOP Count, and correct tensor shapes "
            "for operators. ALWAYS call this after generating the operator list."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "derived_values": {
                    "type": "object",
                    "description": "Derived dimension values for logging",
                },
                "operators": {
                    "type": "array",
                    "description": (
                        "List of operators with Description and initial "
                        "tensor shapes"
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "Description": {"type": "string"},
                            "Count": {"type": "integer"},
                            "FLOP Count": {"type": "integer"},
                            "Input Tensor Shapes": {"type": "string"},
                            "Output Tensor Shapes": {"type": "string"},
                        },
                        "required": ["Description"],
                    },
                },
            },
            "required": ["operators"],
        },
    },
}]


def _build_parallelism_string(cfg: dict) -> str:
    tp = cfg["tensor_parallelism_degree"]
    pp = cfg["pipeline_parallelism_degree"]
    dp = cfg["data_parallelism_degree"]
    dpdcn = cfg.get("data_parallel_degree_dcn", 1)
    tpdcn = cfg.get("tensor_parallel_degree_dcn", 1)
    ppdcn = cfg.get("pipeline_parallel_degree_dcn", 1)
    bs = cfg.get("global_batch_size", 1)
    parts = [f"dp{dp}", f"tp{tp}", f"pp{pp}"]
    if cfg.get("expert_parallelism_degree", 1) > 1:
        parts.append(f"ep{cfg['expert_parallelism_degree']}")
    parts.extend([f"dpdcn{dpdcn}", f"tpdcn{tpdcn}", f"ppdcn{ppdcn}"])
    if cfg.get("expert_parallel_degree_dcn", 1) > 1 or cfg.get("expert_parallelism_degree", 1) > 1:
        parts.append(f"epdcn{cfg.get('expert_parallel_degree_dcn', 1)}")
    parts.append(f"bs{bs}")
    return "-".join(parts)


def _find_best_reference(model_name: str, ffn_type: str, target_par_str: str) -> Path:
    """Find best reference directory for operator description templates.

    Prefers matching tp (highest priority), then pp, then ep.
    tp match matters most because it determines which operators exist
    (e.g., extra AllGather/ReduceScatter ops appear when tp > ep).
    """
    model_dir = _NEUSIM_RESULTS / model_name
    if model_dir.is_dir():
        def _parse_par(name):
            return {m.group(1): int(m.group(2))
                    for m in re.finditer(r'(tp|pp|ep|dp)(\d+)', name)}
        target = _parse_par(target_par_str)
        best, best_score = None, -1
        for sub in sorted(model_dir.iterdir()):
            if not sub.is_dir() or not list(sub.glob("inference-*_prefill.csv")):
                continue
            pars = _parse_par(sub.name)
            score = 0
            if pars.get("tp") == target.get("tp"):
                score += 8
            if pars.get("pp") == target.get("pp"):
                score += 4
            if pars.get("ep") == target.get("ep"):
                score += 2
            if score > best_score:
                best, best_score = sub, score
        if best is not None:
            return best

    # Fallback to a known reference with the same ffn_type
    if ffn_type == "deepseek_moe":
        return (_NEUSIM_RESULTS / "deepseekv2-236b"
                / "dp1-tp16-pp16-ep2-dpdcn1-tpdcn1-ppdcn1-epdcn1-bs16")
    return (_NEUSIM_RESULTS / "llama3_1-405b"
            / "dp1-tp16-pp4-dpdcn1-tpdcn1-ppdcn1-bs16")


def _get_csv_filenames(ref_dir: Path) -> tuple[str, str]:
    prefill = list(ref_dir.glob("inference-*_prefill.csv"))
    decode = list(ref_dir.glob("inference-*_decode.csv"))
    if not prefill or not decode:
        raise FileNotFoundError(f"No reference CSVs in {ref_dir}")
    return prefill[0].name, decode[0].name


def _load_descriptions(path: Path) -> list[str]:
    return list(pd.read_csv(path)["Description"])


# ---------------------------------------------------------------------------
# Post-processing: fix counts and recompute FLOPs from LLM-generated shapes
# ---------------------------------------------------------------------------

def _prime_factorize(n: int) -> list[int]:
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def _split_parallelism(degree: int, num_axes: int) -> list[int]:
    """Split parallelism degree across axes, matching NeuSim's heuristic.

    Takes first num_axes prime factors as initial axes, then distributes
    remaining factors to the smallest axis. May return fewer than num_axes
    elements if degree has fewer prime factors than num_axes.
    """
    if num_axes <= 0 or degree <= 1:
        return [degree] if num_axes >= 1 else []
    if num_axes == 1:
        return [degree]
    factors = _prime_factorize(degree)
    axes = factors[:num_axes]
    remaining = factors[num_axes:]
    for f in remaining:
        min_idx = axes.index(min(axes))
        axes[min_idx] *= f
    return axes


def _get_tp_axes(cfg: dict) -> list[int]:
    """Compute tensor_parallelism_axes mimicking NeuSim's axis mapping logic.

    Assumes v5p (3 physical ICI axes). For v2/v3/v5e/v6e (2 axes), would need
    a chip parameter.
    """
    tp = cfg['tensor_parallelism_degree']
    pp = cfg.get('pipeline_parallelism_degree', 1)
    dp = cfg.get('data_parallelism_degree', 1)
    if tp <= 1:
        return []
    num_physical_axes = 3  # v5p
    tp_en, dp_en, pp_en = tp > 1, dp > 1, pp > 1
    active = sum([tp_en, dp_en, pp_en])
    if active == 1:
        num_tp_axes = num_physical_axes if tp_en else 0
    elif active == 2:
        if tp_en and (dp_en ^ pp_en):
            num_tp_axes = num_physical_axes - 1
        elif not tp_en:
            num_tp_axes = 0
        else:
            num_tp_axes = 1
    elif active == 3:
        num_tp_axes = 1
    else:
        num_tp_axes = 0
    return _split_parallelism(tp, num_tp_axes)


def _compute_rs_flops(input_shape: list[int], tp_axes: list[int], tp: int) -> int:
    """Compute ReduceScatter FLOPs using NeuSim's multi-stage accumulation formula."""
    if tp <= 1:
        return 0
    total_elems = math.prod(input_shape)
    elem_per_chip = math.ceil(total_elems / tp)
    tot_elem = 0
    cur = elem_per_chip
    for dim_size in tp_axes:
        tot_elem += cur
        cur = math.ceil(cur / dim_size)
    return math.ceil(tot_elem * (tp - 1))


def _parse_input_shapes(s: str) -> list[list[int]]:
    """Parse input tensor shapes string into list of dimension lists."""
    if not s or str(s) == 'nan':
        return []
    return [[int(d) for d in m.group(1).split(',')]
            for m in re.finditer(r'\[([\d,]+)\]', str(s))]


def _parse_output_shapes(s: str) -> list[list[int]]:
    """Parse output tensor shapes string into list of dimension lists."""
    if not s or str(s) == 'nan':
        return []
    return [[int(d) for d in m.group(1).split(',')]
            for m in re.finditer(r'\(([\d,]+)\)', str(s))]


def _compute_flops_from_shapes(
    desc: str,
    input_shapes: list[list[int]],
    output_shapes: list[list[int]],
) -> int | None:
    """Compute FLOPs deterministically from parsed shapes and operator type."""
    dl = desc.lower()

    # Zero-FLOP ops: AllGather, AllToAll, ICI Transfer
    zero_kws = ['gather', 'alltoall', 'all to all', 'all_to_all',
                'receive', 'send', 'concatenate']
    if any(kw in dl for kw in zero_kws) and 'reduce' not in dl:
        return 0

    if not output_shapes:
        return 0

    out_shape = output_shapes[0]
    out_prod = math.prod(out_shape)

    # ReduceScatter: FLOPs = product(input) - product(output)
    if 'reduce' in dl and 'scatter' in dl:
        if input_shapes:
            inp_prod = math.prod(input_shapes[0])
            if inp_prod > out_prod:
                return inp_prod - out_prod

    # Unary ops (single input)
    if len(input_shapes) == 1:
        inp_shape = input_shapes[0]
        inp_prod = math.prod(inp_shape)
        if 'rmsnorm' in dl or '_norm' in dl or 'norm' == dl.split('_')[-1]:
            return 6 * inp_prod
        if 'layernorm' in dl:
            return 8 * inp_prod
        if 'softmax' in dl:
            if 'softmax(q' not in dl:
                # Direct softmax (e.g., gate-scores_softmax)
                return 4 * inp_prod
            # Softmax(Q*K)*V sub-ops: the actual softmax has S_full > Nh
            if len(inp_shape) == 4 and inp_shape[2] > inp_shape[3]:
                return 4 * inp_prod
            # Otherwise it's the Add op
            return out_prod
        if 'rotary' in dl:
            return inp_prod
        # Elementwise: Mul (FFgate_up), Add (AttnPlusFFn), combine_output
        return out_prod

    # FlashAttention (3 inputs: Q, K, V)
    if len(input_shapes) >= 3:
        q = input_shapes[0]
        if len(q) == 4:
            B, L, Nh, last_dim = q
            o_last = out_shape[-1] if out_shape else last_dim
            # NeuSim uses d_model=Q_shape[2]=Nh (a quirk), so FLOP includes softmax
            # FLOP = 4 * B * Nh * qL * kvL * d_model + 4 * B * Nh * qL * kvL
            # where d_model = Nh. Simplifies to 4 * B * Nh * L * L * (Nh + 1)
            kv_seqlen = input_shapes[1][1] if len(input_shapes) > 1 else L
            return 4 * B * Nh * L * kv_seqlen * (o_last + 1)

    # Binary ops (2 inputs — matmul / einsum)
    if len(input_shapes) == 2 and output_shapes:
        a_shape = input_shapes[0]
        b_shape = input_shapes[1]
        # Find contracted dimensions using set intersection
        a_set = set(a_shape)
        b_set = set(b_shape)
        o_set = set(out_shape)
        contracted_candidates = (a_set & b_set) - o_set
        if contracted_candidates:
            contracted = math.prod(contracted_candidates)
        else:
            # Fallback: last dim of first input (handles cases where
            # contracted dim has same value as an output dim)
            contracted = a_shape[-1]
        return 2 * out_prod * contracted

    return None


def _fix_rotary_pair(operators: list[dict], cfg: dict, phase: str) -> None:
    """Fix apply_rotary_emb shape swap: Q rotary has Nh dim, KV PE rotary does not."""
    tp = cfg['tensor_parallelism_degree']
    Nh = cfg['num_heads'] // tp
    qk_rope = cfg.get('qk_rope_head_dim', 0)
    if qk_rope == 0:
        return

    L = cfg['input_seqlen'] if phase == 'prefill' else 1
    q_shape = f"DT_FLOAT32:[1,{L},{Nh},{qk_rope}]"
    q_out = f"[DT_FLOAT32:(1,{L},{Nh},{qk_rope})]"
    kv_shape = f"DT_FLOAT32:[1,{L},{qk_rope}]"
    kv_out = f"[DT_FLOAT32:(1,{L},{qk_rope})]"

    rotary_indices = [i for i, op in enumerate(operators)
                      if 'rotary' in op['Description'].lower()]
    if len(rotary_indices) == 2:
        i_q, i_kv = rotary_indices
        operators[i_q]['Input Tensor Shapes'] = q_shape
        operators[i_q]['Output Tensor Shapes'] = q_out
        operators[i_kv]['Input Tensor Shapes'] = kv_shape
        operators[i_kv]['Output Tensor Shapes'] = kv_out


def _fix_llama_shapes(operators: list[dict], cfg: dict, phase: str) -> None:
    """Fix shapes for standard LLM (llama) operators.

    Standard LLM has different patterns from DeepSeek MLA:
    - Q/K/V use 3D weights [d_model, Nh, d_head]
    - Decode KV is combined [2, d_model, Nh, d_head]
    - Decode attention uses prefix/suffix split (6 sub-ops)
    - AllGather MLP is 2D [B, d_model] → [B, d_model*tp]
    - Decode Attention_layernorm uses d_model (not d_model_shard)
    - FFN uses d_ff_chip = d_ff / tp
    """
    tp = cfg['tensor_parallelism_degree']
    pp = cfg['pipeline_parallelism_degree']
    dp = cfg.get('data_parallelism_degree', 1)
    d_model = cfg['d_model']
    d_shard = d_model // tp
    d_ff = cfg['d_ff']
    d_ff_chip = d_ff // tp
    num_heads = cfg['num_heads']
    Nh = num_heads // tp
    d_head = cfg['d_head']
    input_seqlen = cfg['input_seqlen']
    output_seqlen = cfg['output_seqlen']
    bs = cfg.get('global_batch_size', 1)
    B = bs // (dp * pp)
    L = input_seqlen if phase == 'prefill' else 1
    S_prefix = input_seqlen
    S_suffix = output_seqlen
    S_full = S_prefix + S_suffix

    def bf(shape):
        return f"DT_BFLOAT16:[{','.join(str(x) for x in shape)}]"
    def bf_out(shape):
        return f"[DT_BFLOAT16:({','.join(str(x) for x in shape)})]"

    # Track which Softmax(Q*K)*V index we're on (for decode sub-ops)
    qkv_idx = 0
    softmax_idx = 0

    for op in operators:
        dl = op['Description'].lower()

        # AllGather for attention: d_model_shard → d_model (NOT gathered_dim)
        if ('concatenate' in dl or 'allgathermha' in dl or
            'gather results' in dl) and 'mlp' not in dl:
            op['Input Tensor Shapes'] = bf([B, L, d_shard])
            op['Output Tensor Shapes'] = bf_out([B, L, d_model])

        # AllGather for MLP: 2D [B, d_model] → [B, d_model*tp]
        elif 'all gather' in dl and 'mlp' in dl:
            op['Input Tensor Shapes'] = bf([B, d_model])
            op['Output Tensor Shapes'] = bf_out([B, d_model * tp])

        # Q/K/V projections
        elif 'q/k/v' in dl or (dl.startswith('-q-') or dl.startswith('-k-') or dl.startswith('-v-')):
            if phase == 'prefill' or qkv_idx == 0:
                # Q projection (or prefill Q/K/V)
                op['Input Tensor Shapes'] = f"{bf([B, L, d_model])},{bf([d_model, Nh, d_head])}"
                op['Output Tensor Shapes'] = bf_out([B, L, Nh, d_head])
                if 'q/k/v' in dl:
                    qkv_idx += 1
            else:
                # Decode combined KV projection (2nd Q/K/V op)
                op['Input Tensor Shapes'] = f"{bf([B, L, d_model])},{bf([2, d_model, Nh, d_head])}"
                op['Output Tensor Shapes'] = bf_out([B, 2, L, Nh, d_head])
                qkv_idx += 1

        # FlashAttention (prefill only) — NeuSim output uses Nh as last dim (quirk)
        elif 'flashattention' in dl:
            op['Input Tensor Shapes'] = f"{bf([B, L, Nh, d_head])},{bf([B, L, Nh, d_head])},{bf([B, L, Nh, d_head])}"
            op['Output Tensor Shapes'] = bf_out([B, L, Nh, Nh])

        # Decode attention sub-ops: Softmax(Q*K)*V
        elif 'softmax(q*k)*v' in dl or 'softmax(q*k)' in dl:
            if softmax_idx == 0:
                # QK_prefix: BLND;BSND->BLSN
                op['Input Tensor Shapes'] = f"{bf([B, 1, Nh, d_head])},{bf([B, S_prefix, Nh, d_head])}"
                op['Output Tensor Shapes'] = bf_out([B, 1, S_prefix, Nh])
            elif softmax_idx == 1:
                # QK_suffix: BLND;BSND->BLSN
                op['Input Tensor Shapes'] = f"{bf([B, 1, Nh, d_head])},{bf([B, S_suffix, Nh, d_head])}"
                op['Output Tensor Shapes'] = bf_out([B, 1, S_suffix, Nh])
            elif softmax_idx == 2:
                # Softmax over [B,1,S_full,Nh]
                op['Input Tensor Shapes'] = bf([B, 1, S_full, Nh])
                op['Output Tensor Shapes'] = bf_out([B, 1, S_full, Nh])
            elif softmax_idx == 3:
                # attn_avg_prefix: BLSN;BSND->BLND
                op['Input Tensor Shapes'] = f"{bf([B, 1, S_prefix, Nh])},{bf([B, S_prefix, Nh, d_head])}"
                op['Output Tensor Shapes'] = bf_out([B, 1, Nh, d_head])
            elif softmax_idx == 4:
                # attn_avg_suffix: BLSN;BSND->BLND
                op['Input Tensor Shapes'] = f"{bf([B, 1, S_suffix, Nh])},{bf([B, S_suffix, Nh, d_head])}"
                op['Output Tensor Shapes'] = bf_out([B, 1, Nh, d_head])
            elif softmax_idx == 5:
                # Add
                op['Input Tensor Shapes'] = bf([B, 1, Nh, d_head])
                op['Output Tensor Shapes'] = bf_out([B, 1, Nh, d_head])
            softmax_idx += 1

        # Attention output
        elif 'attention_output' in dl:
            op['Input Tensor Shapes'] = f"{bf([B, L, Nh, d_head])},{bf([Nh, d_head, d_model])}"
            op['Output Tensor Shapes'] = bf_out([B, L, d_model])

        # ReduceScatter
        elif 'reduce' in dl and 'scatter' in dl:
            op['Input Tensor Shapes'] = bf([B, L, d_model])
            op['Output Tensor Shapes'] = bf_out([B, L, d_shard])

        # Attention_layernorm
        elif 'attention_layernorm' in dl or ('layernorm' in dl and 'input' not in dl
                                              and 'ffn' not in dl):
            if phase == 'decode':
                # Decode: uses full d_model
                op['Input Tensor Shapes'] = bf([B, L, d_model])
                op['Output Tensor Shapes'] = bf_out([B, L, d_model])
            else:
                # Prefill: uses d_model_shard
                op['Input Tensor Shapes'] = bf([B, L, d_shard])
                op['Output Tensor Shapes'] = bf_out([B, L, d_shard])

        # Input_layernorm / Input RMSNorm
        elif 'input_layernorm' in dl or 'input_rmsnorm' in dl:
            op['Input Tensor Shapes'] = bf([B, L, d_shard])
            op['Output Tensor Shapes'] = bf_out([B, L, d_shard])

        # FFgate_up (Mul) — must check BEFORE FFgate/FFup
        elif 'ffgate_up' in dl:
            op['Input Tensor Shapes'] = bf([B, L, d_ff_chip])
            op['Output Tensor Shapes'] = bf_out([B, L, d_ff_chip])

        # FFgate / FFup
        elif 'ffgate' in dl or 'ffup' in dl:
            op['Input Tensor Shapes'] = f"{bf([B, L, d_model])},{bf([d_model, d_ff_chip])}"
            op['Output Tensor Shapes'] = bf_out([B, L, d_ff_chip])

        # FFoutput
        elif 'ffoutput' in dl:
            op['Input Tensor Shapes'] = f"{bf([B, L, d_ff_chip])},{bf([d_ff_chip, d_model])}"
            op['Output Tensor Shapes'] = bf_out([B, L, d_model])

        # Residual Add (AttnPlusFFn)
        elif 'attnplusffn' in dl or ('add' in dl and 'residual' in dl.replace(' ', '')):
            op['Input Tensor Shapes'] = bf([B, L, d_shard])
            op['Output Tensor Shapes'] = bf_out([B, L, d_shard])


def _fix_ici_shapes(operators: list[dict], cfg: dict, phase: str) -> None:
    """Fix ICI Transfer shapes to use d_model_shard."""
    tp = cfg['tensor_parallelism_degree']
    pp = cfg['pipeline_parallelism_degree']
    dp = cfg.get('data_parallelism_degree', 1)
    B = cfg.get('global_batch_size', 1) // (dp * pp)
    d_shard = cfg['d_model'] // tp
    L = cfg['input_seqlen'] if phase == 'prefill' else 1
    ici_in = f"DT_BFLOAT16:[{B},{L},{d_shard}]"
    ici_out = f"[DT_BFLOAT16:({B},{L},{d_shard})]"

    for op in operators:
        dl = op['Description'].lower()
        if ('receive' in dl or 'send' in dl) and 'ici' in dl:
            op['Input Tensor Shapes'] = ici_in
            op['Output Tensor Shapes'] = ici_out


def _postprocess(operators: list[dict], cfg: dict, phase: str) -> list[dict]:
    """Fix counts, shapes, and recompute FLOPs from shapes.

    The LLM provides correct shapes but sometimes makes arithmetic errors
    in FLOP computation and uses wrong count categories. This function:
    1. Fixes known shape issues (rotary swap, ICI shapes)
    2. Determines count category (ICI / attention / FFN) from operator position
    3. Recomputes FLOPs deterministically from the shape strings
    """
    pp = cfg['pipeline_parallelism_degree']
    num_layers = cfg['num_layers']
    output_seqlen = cfg['output_seqlen']
    decode_width = cfg.get('decode_width', 1)

    layers_per_stage = math.ceil(num_layers / pp)

    ffn_type = cfg.get('ffn_type', 'llama')

    if ffn_type == 'deepseek_moe':
        # DeepSeek MLA: attention ops use num_layers, FFN ops use layers_per_stage
        if phase == 'prefill':
            count_ici = 1
            count_attn = num_layers
            count_ffn = layers_per_stage
        else:
            decode_tokens = output_seqlen // decode_width
            count_ici = decode_tokens
            count_attn = num_layers * decode_tokens
            count_ffn = layers_per_stage * decode_tokens
    else:
        # Standard LLM (llama): ALL per-layer ops use layers_per_stage
        if phase == 'prefill':
            count_ici = 1
            count_attn = layers_per_stage
            count_ffn = layers_per_stage
        else:
            decode_tokens = output_seqlen // decode_width
            count_ici = decode_tokens
            count_attn = layers_per_stage * decode_tokens
            count_ffn = layers_per_stage * decode_tokens

    # Fix known shape issues
    if ffn_type == 'deepseek_moe':
        _fix_rotary_pair(operators, cfg, phase)
    else:
        _fix_llama_shapes(operators, cfg, phase)
    _fix_ici_shapes(operators, cfg, phase)

    # Find AllGatherMHA range — all ops between first and last AllGatherMHA
    # are attention sub-ops
    descs = [op['Description'] for op in operators]
    first_ag = last_ag = None
    for i, d in enumerate(descs):
        if d.lower().startswith('allgathermha') or d.lower().startswith('concatenate'):
            if first_ag is None:
                first_ag = i
            last_ag = i

    for i, op in enumerate(operators):
        dl = op['Description'].lower()

        # Fix Count
        if ('receive' in dl or 'send' in dl) and 'ici' in dl:
            op['Count'] = count_ici
        elif first_ag is not None and first_ag <= i <= last_ag:
            op['Count'] = count_attn
        else:
            op['Count'] = count_ffn

        # Recompute FLOPs from shapes
        inp_shapes = _parse_input_shapes(str(op.get('Input Tensor Shapes', '')))
        out_shapes = _parse_output_shapes(str(op.get('Output Tensor Shapes', '')))

        # ReduceScatter: use NeuSim's multi-stage accumulation formula
        # Matches both "Reduce and scatter" and "Reduce and split"
        if 'reduce' in dl and ('scatter' in dl or 'split' in dl) and inp_shapes:
            tp = cfg['tensor_parallelism_degree']
            tp_axes = _get_tp_axes(cfg)
            op['FLOP Count'] = _compute_rs_flops(inp_shapes[0], tp_axes, tp)
        else:
            flops = _compute_flops_from_shapes(op['Description'], inp_shapes, out_shapes)
            if flops is not None:
                op['FLOP Count'] = flops

    return operators


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = r"""You are a computing demand analyst for neural accelerator simulation.

Given an LLM model config, parallelism setup, and reference operator-description template, you MUST derive ALL numeric values (Count, FLOP Count, tensor shapes) by applying the formulas below step by step. Do NOT guess or approximate — compute exactly.

━━━ OUTPUT FORMAT ━━━
Call the `compute_demand` tool with TWO keys — "derived_values" first, then "operators":
{
  "derived_values": {
    "B": <number>,
    "d_model_shard": <number>,
    "Nh": <number>,
    "d_ff_chip": <number>,
    "layers_per_stage": <number>,
    "decode_tokens": <number>,
    "gathered_dim": <number or null>,
    "q_proj_b_out": <number or null>,
    "wo_input_dim": <number or null>,
    "moe_ff_chip": <number or null>,
    "shared_ff_chip": <number or null>,
    "kv_cache_seq": <number or null>
  },
  "operators": [{"Description": "...", "Count": N, "FLOP Count": N, "Input Tensor Shapes": "...", "Output Tensor Shapes": "..."}, ...]
}

You MUST compute derived_values FIRST from the config, then use ONLY those values in all operators.

━━━ SHAPE FORMAT RULES ━━━
• Input shapes: "DT_BFLOAT16:[d1,d2,...]" — comma-separated for multiple inputs, e.g. "DT_BFLOAT16:[1,1,5120],DT_BFLOAT16:[5120,1536]"
• Output shapes: "[DT_BFLOAT16:(d1,d2,...)]"
• Data types: most ops use DT_BFLOAT16. Only these specific ops use DT_FLOAT32:
    - RMSNorm/LayerNorm (q_norm, kv_norm, but NOT Input_rmsnorm or Attention_RMSNorm which use BF16)
    - Softmax (gate-scores_softmax)
    - Rotary embedding (apply_rotary_emb)
  AllGatherKV, AllGatherK_PE, and all einsums use DT_BFLOAT16.
• Batch dimension B = global_batch_size / (dp × pp). For DeepSeek MoE AllToAll, use global_batch_size.
• Sequence length L: for prefill L = input_seqlen; for decode L = 1.

━━━ DIMENSION FORMULAS ━━━
Compute these from the config BEFORE computing any operator:

  tp = tensor_parallelism_degree
  pp = pipeline_parallelism_degree
  ep = expert_parallelism_degree
  dp = data_parallelism_degree

  B               = global_batch_size / (dp × pp)     (batch per chip)
  d_model_shard   = d_model / tp
  Nh              = num_heads / tp          (per-chip head count — NOT num_heads!)
  Nkv             = num_kv_heads / tp
  d_ff_chip       = d_ff / tp              (FFN dim per chip — MUST divide by tp!)

  For DeepSeek MLA (ffn_type == "deepseek_moe"):
    gathered_dim    = d_model × tp            (MLA AllGatherMHA output — NOT for standard LLM!)
    q_proj_b_out    = Nh × (qk_nope_head_dim + qk_rope_head_dim)     ← uses Nh, NOT num_heads
    wo_input_dim    = Nh × v_head_dim                                  ← uses Nh, NOT num_heads
    moe_ff_chip     = moe_d_ff × ep / tp                               ← MULTIPLY by ep, DIVIDE by tp
    shared_ff_chip  = (d_ff / num_shared_experts) / tp
    kv_cache_seq    = input_seqlen + output_seqlen

━━━ COUNT FORMULAS ━━━

layers_per_stage = CEIL(num_layers / pp)    ← MUST use ceiling division!

Example: ceil(60 / 16) = ceil(3.75) = 4. NOT floor(3.75) = 3.

For STANDARD LLM (ffn_type == "llama"):
  ALL per-layer ops use layers_per_stage. No num_layers distinction.
  PREFILL: ICI = 1, everything else = layers_per_stage
  DECODE:  ICI = decode_tokens, everything else = layers_per_stage × decode_tokens

For DEEPSEEK MoE (ffn_type == "deepseek_moe"):
  Attention sub-ops (AllGatherMHA through AllGatherMHA-post) use num_layers.
  PREFILL: ICI = 1, attention = num_layers, FFN/MoE = layers_per_stage
  DECODE:  ICI = decode_tokens, attention = num_layers × decode_tokens,
           FFN/MoE = layers_per_stage × decode_tokens

decode_tokens = output_seqlen / decode_width

━━━ FLOP COUNT FORMULAS ━━━

• Linear y = x @ W, with x:[B,L,M] W:[M,N]:  FLOPs = 2 × B × L × M × N
• Batched einsum [B,L,Nh,D1] × [B,S,D1] → [B,L,Nh,S]:  FLOPs = 2 × B × L × Nh × D1 × S
• Per-head einsum [B,L,Nh,D1] × [Nh,D2,D1] → [B,L,Nh,D2]:  FLOPs = 2 × B × L × Nh × D1 × D2
• RMSNorm:     FLOPs = 6 × product(input_shape)    ← ALWAYS 6, even for FLOAT32 norms
• LayerNorm:   FLOPs = 8 × product(input_shape)
• Softmax:     FLOPs = 4 × product(input_shape)
• Rotary emb:  FLOPs = product(input_shape)
• Elementwise Mul/Add:  FLOPs = product(output_shape)
• FlashAttention (prefill):  FLOPs = 4 × B × Nh × L × L × d_head
• AllGather, AllToAll, ICI Transfer:  FLOPs = 0
• ReduceScatter:  FLOPs = (tp − 1) × product(output_shape)
• MoE_combine_weights:  FLOPs = 2 × B × L × num_activated_routed_experts_per_token × d_model
• MoE_combine_output:  FLOPs = product(output_shape)

━━━ TENSOR SHAPE PATTERNS ━━━

Standard LLM (ffn_type == "llama"):
  NOTE: Standard LLM AllGather goes d_model_shard → d_model (NOT gathered_dim!).
  NOTE: All norms in llama use LayerNorm (8× FLOPs), NOT RMSNorm (6×).
  NOTE: ALL per-layer ops use layers_per_stage count (no num_layers distinction).
  NOTE: Q/K/V weights are 3D [d_model, Nh, d_head], NOT 2D [d_model, Nh*d_head].
  NOTE: FFN uses d_ff_chip = d_ff / tp. MUST divide by tp!

  S_prefix = input_seqlen, S_suffix = output_seqlen, S_full = S_prefix + S_suffix

  PREFILL (flash attention path):
    ICI Receive:              in=BF16:[B,L,d_model_shard]                     out=[B,L,d_model_shard]
    Input_layernorm:          in=BF16:[B,L,d_model_shard]                     out=[B,L,d_model_shard]
    AllGather (Concatenate):  in=BF16:[B,L,d_model_shard]                     out=[B,L,d_model]
    Q (BLM;MND→BLND):        in=BF16:[B,L,d_model],[d_model,Nh,d_head]       out=[B,L,Nh,d_head]
    K (BLM;MND→BLND):        in=BF16:[B,L,d_model],[d_model,Nh,d_head]       out=[B,L,Nh,d_head]
    V (BLM;MND→BLND):        in=BF16:[B,L,d_model],[d_model,Nh,d_head]       out=[B,L,Nh,d_head]
    FlashAttention:           in=BF16:[B,L,Nh,d_head]×3                       out=[B,L,Nh,d_head]
    Attn output (BLND;NDM→BLM): in=BF16:[B,L,Nh,d_head],[Nh,d_head,d_model]  out=[B,L,d_model]
    ReduceScatter:            in=BF16:[B,L,d_model]                           out=[B,L,d_model_shard]
    Attention_layernorm:      in=BF16:[B,L,d_model_shard]                     out=[B,L,d_model_shard]
    AllGather→LN:             in=BF16:[B,L,d_model_shard]                     out=[B,L,d_model]
    AllGather→MLP (2D!):      in=BF16:[B,d_model]                             out=[B,d_model×tp]
    FFgate/FFup:              in=BF16:[B,L,d_model],[d_model,d_ff_chip]       out=[B,L,d_ff_chip]
    FFgate_up (Mul):          in=BF16:[B,L,d_ff_chip]                         out=[B,L,d_ff_chip]
    FFoutput:                 in=BF16:[B,L,d_ff_chip],[d_ff_chip,d_model]     out=[B,L,d_model]
    ReduceScatter-FFN:        in=BF16:[B,L,d_model]                           out=[B,L,d_model_shard]
    Residual Add:             in=BF16:[B,L,d_model_shard]                     out=[B,L,d_model_shard]

  DECODE (self-attention, prefix/suffix split):
    ICI Receive:              in=BF16:[B,1,d_model_shard]                     out=[B,1,d_model_shard]
    Input_layernorm:          in=BF16:[B,1,d_model_shard]                     out=[B,1,d_model_shard]
    AllGatherMHA:             in=BF16:[B,1,d_model_shard]                     out=[B,1,d_model]
    Q/K/V (1st = Q, BLM;MND→BLND):   in=BF16:[B,1,d_model],[d_model,Nh,d_head]     out=[B,1,Nh,d_head]
    Q/K/V (2nd = KV, BLM;TMND→BTLND): in=BF16:[B,1,d_model],[2,d_model,Nh,d_head]  out=[B,2,1,Nh,d_head]
    Softmax(Q*K)*V #1 (QK_prefix):   in=BF16:[B,1,Nh,d_head],[B,S_prefix,Nh,d_head]  out=[B,1,S_prefix,Nh]
    Softmax(Q*K)*V #2 (QK_suffix):   in=BF16:[B,1,Nh,d_head],[B,S_suffix,Nh,d_head]  out=[B,1,S_suffix,Nh]
    Softmax(Q*K)*V #3 (Softmax):     in=BF16:[B,1,S_full,Nh]                          out=[B,1,S_full,Nh]
    Softmax(Q*K)*V #4 (attn_prefix): in=BF16:[B,1,S_prefix,Nh],[B,S_prefix,Nh,d_head] out=[B,1,Nh,d_head]
    Softmax(Q*K)*V #5 (attn_suffix): in=BF16:[B,1,S_suffix,Nh],[B,S_suffix,Nh,d_head] out=[B,1,Nh,d_head]
    Softmax(Q*K)*V #6 (Add):         in=BF16:[B,1,Nh,d_head]                          out=[B,1,Nh,d_head]
    Attn output (BLND;NDM→BLM): in=BF16:[B,1,Nh,d_head],[Nh,d_head,d_model]  out=[B,1,d_model]
    ReduceScatter:            in=BF16:[B,1,d_model]                           out=[B,1,d_model_shard]
    Attention_layernorm:      in=BF16:[B,1,d_model]                           out=[B,1,d_model]
        ↑ IMPORTANT: decode Attention_layernorm uses d_model (FULL), NOT d_model_shard!
    AllGatherMHA-post:        in=BF16:[B,1,d_model_shard]                     out=[B,1,d_model]
    AllGather→MLP (2D!):      in=BF16:[B,d_model]                             out=[B,d_model×tp]
    FFgate/FFup:              in=BF16:[B,1,d_model],[d_model,d_ff_chip]       out=[B,1,d_ff_chip]
    FFgate_up (Mul):          in=BF16:[B,1,d_ff_chip]                         out=[B,1,d_ff_chip]
    FFoutput:                 in=BF16:[B,1,d_ff_chip],[d_ff_chip,d_model]     out=[B,1,d_model]
    ReduceScatter-FFN:        in=BF16:[B,1,d_model]                           out=[B,1,d_model_shard]
    Residual Add:             in=BF16:[B,1,d_model_shard]                     out=[B,1,d_model_shard]

DeepSeek MLA (ffn_type == "deepseek_moe"):
  NOTE: For MLA, AllGatherMHA gathers d_model → gathered_dim (= d_model×tp).
  This is DIFFERENT from standard LLM where AllGather goes d_model_shard → d_model.

  ICI Transfer:                in=BF16:[1,L,d_model_shard]                     out=BF16:[1,L,d_model_shard]
  Input RMSNorm:               in=BF16:[1,L,d_model_shard]                     out=BF16:[1,L,d_model_shard]
  AllGatherMHA:                in=BF16:[1,L,d_model]                            out=BF16:[1,L,gathered_dim]
      INPUT is d_model (NOT d_model_shard). OUTPUT is gathered_dim = d_model × tp.
  q_wq_a_linear:               in=BF16:[1,L,d_model],[d_model,q_lora_rank]     out=BF16:[1,L,q_lora_rank]
  q_norm:                      in=F32:[1,L,q_lora_rank]                        out=F32:[1,L,q_lora_rank]
  q_wq_b_linear:               in=BF16:[1,L,q_lora_rank],[q_lora_rank,q_proj_b_out]  out=BF16:[1,L,q_proj_b_out]
  apply_rotary_emb (q):        in=F32:[1,L,Nh,qk_rope_head_dim]                out=F32:[1,L,Nh,qk_rope_head_dim]
  kv_k_pe_linear:              in=BF16:[1,L,d_model],[d_model,kv_lora_rank+qk_rope_head_dim]
                               out=BF16:[1,L,kv_lora_rank+qk_rope_head_dim]
  apply_rotary_emb (kv_pe):    in=F32:[1,L,qk_rope_head_dim]                   out=F32:[1,L,qk_rope_head_dim]
  AllGatherKV:                 in=BF16:[1,L,kv_lora_rank]                       out=BF16:[1,L×tp,kv_lora_rank]
      (gathers along seq dim; for decode L=1 → output=[1,tp,kv_lora_rank])
  AllGatherK_PE:               in=BF16:[1,L,qk_rope_head_dim]                  out=BF16:[1,L×tp,qk_rope_head_dim]
  q_nope_wkv_b_linear:         in=BF16:[1,L,Nh,qk_nope_head_dim],[Nh,qk_nope_head_dim,kv_lora_rank]
                               out=BF16:[1,L,Nh,kv_lora_rank]
  kv_norm:                     in=F32:[1,L,kv_lora_rank]                       out=F32:[1,L,kv_lora_rank]
  q_nope_kv_cache_einsum:      in=BF16:[1,L,Nh,kv_lora_rank],[1,S,kv_lora_rank]  out=BF16:[1,L,Nh,S]
      S = L (prefill) or kv_cache_seq (decode)
  q_pe_pe_einsum:              in=BF16:[1,L,Nh,qk_rope_head_dim],[1,S,qk_rope_head_dim]  out=BF16:[1,L,Nh,S]
  scores_kv_cache_einsum:      in=BF16:[1,L,Nh,S],[1,S,kv_lora_rank]           out=BF16:[1,L,Nh,kv_lora_rank]
  x_wkv_b_linear:              in=BF16:[1,L,Nh,kv_lora_rank],[Nh,v_head_dim,kv_lora_rank]
                               out=BF16:[1,L,Nh,v_head_dim]
  x_wo_linear:                 in=BF16:[1,L,wo_input_dim],[wo_input_dim,d_model]  out=BF16:[1,L,d_model]
      wo_input_dim = Nh × v_head_dim
  ReduceScatter-attn:          in=BF16:[1,L,gathered_dim]                       out=BF16:[1,L,d_model]
  Attention_RMSNorm:           in=BF16:[1,L,gathered_dim]                       out=BF16:[1,L,gathered_dim]
  AllGatherMHA-post:           in=BF16:[1,L,d_model]                            out=BF16:[1,L,gathered_dim]
  MoE gate-scores:             in=BF16:[1,L,d_model],[num_routed_experts,d_model]  out=BF16:[1,L,num_routed_experts]
  gate-scores_softmax:         in=F32:[1,L,num_routed_experts]                  out=F32:[1,L,num_routed_experts]
  MoE_Dispatch_AllToAll:       in=BF16:[global_batch_size,L,num_limited_groups,d_model]
                               out=BF16:[global_batch_size,L,num_limited_groups,d_model]
      (AllToAll uses global_batch_size as first dim, NOT 1)
  Routed expert FFgate/FFup:   in=BF16:[1,L,d_model],[d_model,moe_ff_chip]     out=BF16:[1,L,moe_ff_chip]
  Routed expert FFgate_up:     in=BF16:[1,L,moe_ff_chip]                        out=BF16:[1,L,moe_ff_chip]
  Routed expert FFoutput:      in=BF16:[1,L,moe_ff_chip],[moe_ff_chip,d_model]  out=BF16:[1,L,d_model]
  Shared expert FFgate/FFup:   in=BF16:[1,L,d_model],[d_model,shared_ff_chip]   out=BF16:[1,L,shared_ff_chip]
  Shared expert FFgate_up:     in=BF16:[1,L,shared_ff_chip]                      out=BF16:[1,L,shared_ff_chip]
  Shared expert FFoutput:      in=BF16:[1,L,shared_ff_chip],[shared_ff_chip,d_model]  out=BF16:[1,L,d_model]
  MoE_combine_weights:         in=BF16:[1,L,num_activated,d_model],[1,L,num_activated]  out=BF16:[1,L,d_model]
  MoE_combine_output:          in=BF16:[1,L,d_model]                            out=BF16:[1,L,d_model]
  Residual Add:                in=BF16:[1,L,d_model]                            out=BF16:[1,L,d_model]
      (MoE models: residual uses d_model, NOT d_model_shard)

━━━ COMMON MISTAKES — READ THESE BEFORE COMPUTING ━━━

1. layers_per_stage = CEIL(num_layers / pp), NOT floor.
   ceil(60/16) = 4. NOT 3.

2. Attention sub-ops (AllGatherMHA through AllGatherMHA-post) count = num_layers × decode_tokens.
   Everything else (Input_rmsnorm, FFN, MoE, residual) count = layers_per_stage × decode_tokens.

3. For MLA: AllGatherMHA input is d_model (NOT d_model_shard), output is gathered_dim = d_model × tp.
   This differs from standard LLM where AllGather goes d_model_shard → d_model.

4. q_proj_b_out = Nh × (qk_nope_head_dim + qk_rope_head_dim). Uses Nh = num_heads/tp.

5. wo_input_dim = Nh × v_head_dim. Uses Nh = num_heads/tp.

6. moe_ff_chip = moe_d_ff × ep / tp. MULTIPLY by ep, then DIVIDE by tp.

7. shared_ff_chip = (d_ff / num_shared_experts) / tp.

8. RMSNorm FLOPs = 6 × product(input_shape) ALWAYS (even for FLOAT32 norms).

9. Softmax FLOPs = 4 × product(input_shape).

10. MoE_combine_weights FLOPs = 2 × B × L × num_activated × d_model. (factor of 2!)

11. Use comma (,) between multiple input tensors. NOT × or semicolons.

12. gate-scores weight shape is [num_routed_experts, d_model], NOT [d_model, num_routed_experts].
    FLOPs = 2 × B × L × d_model × num_routed_experts.

━━━ TOOL USAGE ━━━

You have access to a `compute_demand` tool. After generating the operator list with
descriptions and initial tensor shapes, you MUST call this tool. The tool will:
- Compute precise Count values based on operator position and model config
- Compute exact FLOP Count from tensor shapes using deterministic formulas
- Correct any tensor shape errors

Pass ALL operators to the tool in a single call. The tool returns the finalized operator list.

━━━ FINAL INSTRUCTIONS ━━━
1. First compute ALL derived_values.
2. Then use ONLY those derived values to compute each operator.
3. Use the reference descriptions EXACTLY (same strings, same order).
4. Before calling the tool, verify: did Nh come from num_heads/tp? Did gathered_dim come from d_model×tp?
5. Call the `compute_demand` tool with derived_values and operators.
"""


def _build_user_prompt(cfg: dict, phase: str, ref_descriptions: list[str]) -> str:
    desc_list = "\n".join(f"  {i+1}. {d}" for i, d in enumerate(ref_descriptions))

    seq_len = cfg['input_seqlen'] if phase == 'prefill' else 1

    return f"""Compute the per-operator computing demand for the **{phase}** phase.

## Model Configuration
```json
{json.dumps(cfg, indent=2)}
```

## Key Parameters
tp={cfg['tensor_parallelism_degree']}, pp={cfg['pipeline_parallelism_degree']}, dp={cfg['data_parallelism_degree']}, ep={cfg.get('expert_parallelism_degree',1)}, bs={cfg['global_batch_size']}
input_seqlen={cfg['input_seqlen']}, output_seqlen={cfg['output_seqlen']}, decode_width={cfg.get('decode_width',1)}

## Phase
This is the **{phase}** phase. Sequence length L = {seq_len}. Batch = 1 (always).

## IMPORTANT: Compute derived_values FIRST
Use the dimension formulas from the system prompt to compute ALL derived_values and include them in your JSON response BEFORE computing operators. Use ONLY those derived values for all operators.

## Reference Operator Descriptions (use these EXACT strings, in this EXACT order)
{desc_list}

Call the compute_demand tool with "derived_values" and "operators". All {len(ref_descriptions)} operators must be included."""


class ComputeDemandAgent:
    """Agent that uses OpenAI to compute per-operator demand from model configs."""

    def __init__(
        self,
        model_config_path: str,
        tp: int | None = None,
        pp: int | None = None,
        dp: int | None = None,
        ep: int | None = None,
        bs: int | None = None,
        dpdcn: int | None = None,
        tpdcn: int | None = None,
        ppdcn: int | None = None,
        epdcn: int | None = None,
        output_dir: str | None = None,
    ):
        self.config_path = Path(model_config_path)
        with open(self.config_path) as f:
            self.config = json.load(f)

        # Apply overrides
        overrides = {
            "tensor_parallelism_degree": tp,
            "pipeline_parallelism_degree": pp,
            "data_parallelism_degree": dp,
            "expert_parallelism_degree": ep,
            "global_batch_size": bs,
            "data_parallel_degree_dcn": dpdcn,
            "tensor_parallel_degree_dcn": tpdcn,
            "pipeline_parallel_degree_dcn": ppdcn,
            "expert_parallel_degree_dcn": epdcn,
        }
        for key, val in overrides.items():
            if val is not None:
                self.config[key] = val

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = (
                Path(__file__).resolve().parent / "results"
                / self.config["model_name"]
                / _build_parallelism_string(self.config)
            )

        self.ffn_type = self.config.get("ffn_type", "llama")

        # Find best reference for operator description template
        par_str = _build_parallelism_string(self.config)
        self.ref_dir = _find_best_reference(self.config["model_name"], self.ffn_type, par_str)
        self.prefill_csv, self.decode_csv = _get_csv_filenames(self.ref_dir)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment or .env file")
        self.client = OpenAI(api_key=api_key)

    def _execute_tool(self, tool_name: str, arguments: dict,
                      ref_descriptions: list[str], phase: str) -> list[dict]:
        """Execute a tool call and return results."""
        if tool_name == "compute_demand":
            operators = arguments["operators"]
            # Log derived values if provided
            dv = arguments.get("derived_values", {})
            if dv:
                print(f"  Derived values: Nh={dv.get('Nh')}, "
                      f"gathered_dim={dv.get('gathered_dim')}, "
                      f"moe_ff_chip={dv.get('moe_ff_chip')}, "
                      f"shared_ff_chip={dv.get('shared_ff_chip')}, "
                      f"q_proj_b_out={dv.get('q_proj_b_out')}, "
                      f"wo_input_dim={dv.get('wo_input_dim')}, "
                      f"layers_per_stage={dv.get('layers_per_stage')}")
            operators = self._validate(operators, ref_descriptions, phase)
            operators = _postprocess(operators, self.config, phase)
            print(f"  Tool computed {len(operators)} operators for {phase}")
            return operators
        raise ValueError(f"Unknown tool: {tool_name}")

    def _call_openai(self, phase: str, ref_descriptions: list[str]) -> list[dict]:
        user_prompt = _build_user_prompt(self.config, phase, ref_descriptions)

        print(f"  Calling OpenAI for {phase} phase...")
        response = self.client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            tools=TOOLS,
            tool_choice={"type": "function", "function": {"name": "compute_demand"}},
            max_completion_tokens=64000,
        )

        message = response.choices[0].message
        if message.tool_calls:
            tc = message.tool_calls[0]
            args = json.loads(tc.function.arguments)
            return self._execute_tool(
                tc.function.name, args, ref_descriptions, phase)

        # Fallback: no tool call (shouldn't happen with tool_choice)
        content = message.content or "{}"
        result = json.loads(content)
        operators = result.get("operators", [])
        operators = self._validate(operators, ref_descriptions, phase)
        operators = _postprocess(operators, self.config, phase)
        return operators

    def _validate(self, operators: list[dict], ref_descs: list[str], phase: str) -> list[dict]:
        for op in operators:
            for col in OUTPUT_COLUMNS:
                if col not in op:
                    op[col] = 0 if col in ("Count", "FLOP Count") else ""
            op["Count"] = int(op["Count"])
            op["FLOP Count"] = int(op["FLOP Count"])

        gen_descs = [o["Description"] for o in operators]
        missing = set(ref_descs) - set(gen_descs)
        extra = set(gen_descs) - set(ref_descs)
        if missing:
            print(f"  WARNING [{phase}]: Missing operators: {missing}")
        if extra:
            print(f"  WARNING [{phase}]: Extra operators: {extra}")
        return operators

    def _save_csv(self, operators: list[dict], filename: str) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / filename
        pd.DataFrame(operators, columns=OUTPUT_COLUMNS).to_csv(path, index=False)
        print(f"  Saved: {path}")
        return path

    def run(self) -> tuple[Path, Path]:
        cfg = self.config
        print(f"Generating computing demand for: {cfg['model_name']}")
        print(f"  FFN type: {self.ffn_type}")
        print(f"  TP={cfg['tensor_parallelism_degree']}, "
              f"PP={cfg['pipeline_parallelism_degree']}, "
              f"DP={cfg['data_parallelism_degree']}, "
              f"EP={cfg.get('expert_parallelism_degree', 1)}, "
              f"BS={cfg['global_batch_size']}")
        print(f"  Reference template: {self.ref_dir.name}")
        print(f"  Output dir: {self.output_dir}")

        pref_descs = _load_descriptions(self.ref_dir / self.prefill_csv)
        dec_descs = _load_descriptions(self.ref_dir / self.decode_csv)

        prefill_ops = self._call_openai("prefill", pref_descs)
        prefill_path = self._save_csv(prefill_ops, "inference_prefill.csv")

        decode_ops = self._call_openai("decode", dec_descs)
        decode_path = self._save_csv(decode_ops, "inference_decode.csv")

        print("Done!")
        return prefill_path, decode_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-operator computing demand CSVs for LLM models"
    )
    parser.add_argument("model_config", help="Path to model config JSON file")
    parser.add_argument("--tp", type=int, default=None)
    parser.add_argument("--pp", type=int, default=None)
    parser.add_argument("--dp", type=int, default=None)
    parser.add_argument("--ep", type=int, default=None)
    parser.add_argument("--bs", type=int, default=None)
    parser.add_argument("--dpdcn", type=int, default=None)
    parser.add_argument("--tpdcn", type=int, default=None)
    parser.add_argument("--ppdcn", type=int, default=None)
    parser.add_argument("--epdcn", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    agent = ComputeDemandAgent(
        model_config_path=args.model_config,
        tp=args.tp, pp=args.pp, dp=args.dp, ep=args.ep, bs=args.bs,
        dpdcn=args.dpdcn, tpdcn=args.tpdcn, ppdcn=args.ppdcn, epdcn=args.epdcn,
        output_dir=args.output_dir,
    )
    prefill_path, decode_path = agent.run()
    print(f"\nPrefill CSV: {prefill_path}")
    print(f"Decode CSV: {decode_path}")


if __name__ == "__main__":
    main()
