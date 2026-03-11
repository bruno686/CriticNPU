"""Cross-Workload Design Memory: stores and retrieves NPU design experiences.

Enables experience transfer across workloads by maintaining two memory tiers:

  Episodic Memory: specific design runs with configs, results, and LLM-generated reflections
  Semantic Memory: general design principles consolidated from multiple episodes

Integration point in the design loop:
  1. Before design starts → retrieve relevant experiences → inject into LLM context
  2. After design completes → reflect on trajectory → store episode
  3. Periodically → consolidate episodes into semantic rules
"""

import json
import math
from pathlib import Path
from typing import Any

from openai import OpenAI


# ── Workload Feature Extraction ──────────────────────────────────────────────

def extract_workload_features(
    model_config: dict[str, Any],
    tp: int,
    pp: int,
    bs: int,
) -> dict[str, float]:
    """Compute per-chip workload features from model config and parallelism.

    These features capture *what each chip actually sees*, which determines
    the optimal NPU config. Two workloads with similar per-chip features
    should have similar optimal configs.
    """
    d_model = model_config.get("d_model", 4096)
    d_ff = model_config.get("d_ff", d_model * 4)
    num_heads = model_config.get("num_heads", 32)
    num_kv_heads = model_config.get("num_kv_heads", num_heads)
    num_layers = model_config.get("num_layers", 32)
    input_seqlen = model_config.get("input_seqlen", 4096)
    output_seqlen = model_config.get("output_seqlen", 512)
    d_head = model_config.get("d_head", d_model // num_heads)
    ffn_type = model_config.get("ffn_type", "dense")

    # Per-chip dimensions (after tensor parallelism sharding)
    per_chip_d_model = d_model  # d_model is not sharded
    per_chip_d_ff = d_ff / tp
    per_chip_heads = num_heads / tp
    per_chip_kv_heads = max(1, num_kv_heads / tp)
    per_chip_layers = num_layers / pp

    # Rough per-chip FLOP estimate (prefill, one forward pass)
    # Attention: 4 * seq * d_model * d_model (Q/K/V/O projections) + 2 * seq^2 * d_model (attn)
    attn_flops = (4 * input_seqlen * per_chip_d_model * (per_chip_heads * d_head)
                  + 2 * input_seqlen * input_seqlen * (per_chip_heads * d_head))
    # FFN: 2 * seq * d_model * d_ff (up + down projections; llama has gate too)
    ffn_multiplier = 3 if ffn_type == "llama" else 2
    ffn_flops = ffn_multiplier * input_seqlen * per_chip_d_model * per_chip_d_ff
    total_flops = (attn_flops + ffn_flops) * per_chip_layers * bs

    # Rough per-chip memory traffic estimate (weight load)
    # Attention weights: 4 * d_model * (heads * d_head) * 2 bytes
    attn_bytes = 4 * per_chip_d_model * (per_chip_heads * d_head) * 2
    ffn_bytes = ffn_multiplier * per_chip_d_model * per_chip_d_ff * 2
    total_bytes = (attn_bytes + ffn_bytes) * per_chip_layers

    # Arithmetic intensity (FLOP / byte) — key indicator of compute vs memory bound
    arithmetic_intensity = total_flops / total_bytes if total_bytes > 0 else 0

    # Is MoE?
    is_moe = 1.0 if ffn_type in ("deepseek", "moe") or model_config.get("num_routed_experts", 0) > 0 else 0.0

    return {
        "per_chip_d_ff": per_chip_d_ff,
        "per_chip_heads": per_chip_heads,
        "per_chip_kv_heads": per_chip_kv_heads,
        "per_chip_layers": per_chip_layers,
        "d_head": d_head,
        "input_seqlen": input_seqlen,
        "output_seqlen": output_seqlen,
        "batch_size": bs,
        "arithmetic_intensity": arithmetic_intensity,
        "total_flops_estimate": total_flops,
        "total_bytes_estimate": total_bytes,
        "is_moe": is_moe,
    }


def workload_similarity(feat_a: dict[str, float], feat_b: dict[str, float]) -> float:
    """Cosine similarity between two workload feature vectors.

    Returns a value in [0, 1] where 1 = identical per-chip workload.
    """
    # Use log-scale for large-range features to avoid magnitude domination
    log_keys = {"per_chip_d_ff", "total_flops_estimate", "total_bytes_estimate",
                "input_seqlen", "output_seqlen", "arithmetic_intensity"}

    keys = sorted(set(feat_a.keys()) & set(feat_b.keys()))
    vec_a, vec_b = [], []
    for k in keys:
        a, b = feat_a[k], feat_b[k]
        if k in log_keys:
            a = math.log1p(abs(a))
            b = math.log1p(abs(b))
        vec_a.append(a)
        vec_b.append(b)

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── Reflection (Self-Reflection after design run) ───────────────────────────

REFLECTION_PROMPT = """You are an NPU design analyst. Given a completed design trajectory, write a concise design lesson in 2-3 sentences.

Focus on:
- What workload characteristic drove the optimal config (compute-bound → more SAs? memory-bound → more HBM BW?)
- Any counterintuitive finding (e.g., lower frequency being better)
- The key trade-off that determined the final config

Do NOT list parameter values. State the PRINCIPLE, not the numbers.

Example:
"For this compute-bound prefill workload, the optimal strategy was maximizing SA count while using the lowest viable frequency to stay within power budget. The power savings from low frequency funded 2 additional SAs, which improved TTFT more than higher clock speed would have."
"""


def generate_reflection(
    model_name: str,
    optimization_target: str,
    best_config: dict[str, Any],
    best_results: dict[str, Any],
    trajectory_summary: str,
    client: OpenAI | None = None,
) -> str:
    """Use LLM to generate a design lesson from a completed trajectory."""
    if client is None:
        client = OpenAI()

    user_content = (
        f"Model: {model_name}\n"
        f"Optimization: {optimization_target}\n"
        f"Best config: {json.dumps(best_config)}\n"
        f"Best results: TTFT={best_results.get('TTFT_sec', 'N/A')}s, "
        f"TPOT={best_results.get('TPOT_ms_request', 'N/A')}ms, "
        f"Power={best_results.get('total_power_peak_W', 'N/A')}W\n"
        f"\nDesign trajectory:\n{trajectory_summary}"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": REFLECTION_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
        max_tokens=200,
    )
    return response.choices[0].message.content


def _summarize_trajectory(history: list[dict]) -> str:
    """Create a compact trajectory summary for reflection."""
    lines = []
    for h in history:
        r = h.get("results", {})
        valid = not r.get("out_of_memory", True) and r.get("total_power_peak_W", 999) <= 300
        cfg = h.get("chip_overrides", {})
        key_params = {k: cfg[k] for k in ("num_sa", "sa_dim", "freq_GHz", "vmem_size_MB") if k in cfg}
        lines.append(
            f"  iter {h['iteration']}: {json.dumps(key_params)} → "
            f"TTFT={r.get('TTFT_sec', 'N/A'):.4f}s, TPOT={r.get('TPOT_ms_request', 'N/A'):.4f}ms, "
            f"Pwr={r.get('total_power_peak_W', 0):.0f}W {'✓' if valid else '✗'}"
        )
    return "\n".join(lines)


# ── Consolidation (Episodic → Semantic) ──────────────────────────────────────

CONSOLIDATION_PROMPT = """You are an NPU design expert. Given multiple design experiences across different workloads, extract 2-3 GENERAL design principles that apply broadly.

Rules:
- State principles, not workload-specific facts
- Each principle should be actionable for a new design
- Be concise: one sentence per principle

Example output:
- "For memory-bound workloads, increasing HBM bandwidth yields diminishing returns past 80% utilization — shift to larger on-chip SRAM instead."
- "Lowering core frequency below 0.7 GHz frees enough power budget for 2-3 additional SAs, which improves throughput for compute-bound workloads more than higher clock speed."
"""


def consolidate_episodes(
    episodes: list[dict[str, Any]],
    client: OpenAI | None = None,
) -> list[str]:
    """Consolidate multiple episodic memories into semantic principles."""
    if len(episodes) < 2:
        return []

    if client is None:
        client = OpenAI()

    episode_descriptions = []
    for ep in episodes:
        episode_descriptions.append(
            f"- {ep['model_name']} ({ep['optimization_target']}): {ep['reflection']}"
        )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": CONSOLIDATION_PROMPT},
            {"role": "user", "content": (
                "Design experiences:\n" + "\n".join(episode_descriptions) +
                "\n\nExtract general principles (one per line, starting with '- ')."
            )},
        ],
        temperature=0.2,
        max_tokens=300,
    )

    text = response.choices[0].message.content
    principles = [line.strip().lstrip("- ") for line in text.strip().split("\n") if line.strip().startswith("-")]
    return principles


# ── Memory Store ─────────────────────────────────────────────────────────────

class DesignMemory:
    """Persistent cross-workload design memory with episodic and semantic tiers.

    Storage layout (JSON files in memory_dir/):
        episodes.json   — list of episodic memories
        semantics.json  — list of consolidated design principles
    """

    def __init__(self, memory_dir: str = "design_memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._episodes_path = self.memory_dir / "episodes.json"
        self._semantics_path = self.memory_dir / "semantics.json"

    # ── Load / Save ──

    def _load_episodes(self) -> list[dict[str, Any]]:
        if self._episodes_path.exists():
            return json.loads(self._episodes_path.read_text())
        return []

    def _save_episodes(self, episodes: list[dict[str, Any]]) -> None:
        self._episodes_path.write_text(json.dumps(episodes, indent=2, default=str))

    def _load_semantics(self) -> list[str]:
        if self._semantics_path.exists():
            return json.loads(self._semantics_path.read_text())
        return []

    def _save_semantics(self, semantics: list[str]) -> None:
        self._semantics_path.write_text(json.dumps(semantics, indent=2))

    # ── Store ──

    def store_episode(
        self,
        model_name: str,
        optimization_target: str,
        workload_features: dict[str, float],
        best_config: dict[str, Any],
        best_results: dict[str, Any],
        parallelism: dict[str, int],
        reflection: str,
        power_budget_W: float,
    ) -> None:
        """Store a completed design run as an episodic memory."""
        episode = {
            "model_name": model_name,
            "optimization_target": optimization_target,
            "workload_features": workload_features,
            "best_config": best_config,
            "best_results": {
                "TTFT_sec": best_results.get("TTFT_sec"),
                "TPOT_ms_request": best_results.get("TPOT_ms_request"),
                "total_power_peak_W": best_results.get("total_power_peak_W"),
            },
            "parallelism": parallelism,
            "power_budget_W": power_budget_W,
            "reflection": reflection,
        }
        episodes = self._load_episodes()
        episodes.append(episode)
        self._save_episodes(episodes)

    # ── Retrieve ──

    def retrieve(
        self,
        workload_features: dict[str, float],
        optimization_target: str,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Retrieve relevant design experiences for a new workload.

        Returns a dict with:
            - episodic: list of most similar episodes (up to top_k)
            - semantic: list of general design principles
        """
        episodes = self._load_episodes()
        semantics = self._load_semantics()

        # Filter episodes by optimization target (prefill vs decode)
        target_episodes = [
            ep for ep in episodes
            if ep["optimization_target"] == optimization_target
        ]

        # Rank by workload similarity
        scored = []
        for ep in target_episodes:
            sim = workload_similarity(workload_features, ep["workload_features"])
            scored.append((sim, ep))
        scored.sort(key=lambda x: x[0], reverse=True)

        return {
            "episodic": [(sim, ep) for sim, ep in scored[:top_k]],
            "semantic": semantics,
        }

    # ── Consolidate ──

    def consolidate(self, client: OpenAI | None = None) -> list[str]:
        """Consolidate all episodes into semantic principles. Replaces existing semantics."""
        episodes = self._load_episodes()
        if len(episodes) < 2:
            return self._load_semantics()

        principles = consolidate_episodes(episodes, client=client)
        self._save_semantics(principles)
        return principles

    # ── Format for LLM Context ──

    def format_for_context(
        self,
        workload_features: dict[str, float],
        optimization_target: str,
        top_k: int = 3,
    ) -> str:
        """Retrieve and format relevant experiences for injection into LLM context.

        Returns a formatted string ready to be added to the user message,
        or empty string if no relevant memories exist.
        """
        retrieved = self.retrieve(workload_features, optimization_target, top_k=top_k)

        if not retrieved["episodic"] and not retrieved["semantic"]:
            return ""

        lines = ["## Design Experience from Previous Workloads"]

        # Semantic principles first (general knowledge)
        if retrieved["semantic"]:
            lines.append("\nGeneral design principles learned:")
            for principle in retrieved["semantic"]:
                lines.append(f"- {principle}")

        # Episodic memories (specific similar experiences)
        if retrieved["episodic"]:
            lines.append("\nSimilar workload design experiences:")
            for sim, ep in retrieved["episodic"]:
                config_summary = {
                    k: ep["best_config"][k]
                    for k in ("num_sa", "sa_dim", "freq_GHz", "num_vu", "vmem_size_MB")
                    if k in ep["best_config"]
                }
                r = ep["best_results"]
                lines.append(
                    f"- {ep['model_name']} ({ep['optimization_target']}, "
                    f"similarity={sim:.2f}): "
                    f"best config={json.dumps(config_summary)}, "
                    f"TTFT={r.get('TTFT_sec', 'N/A')}s, "
                    f"TPOT={r.get('TPOT_ms_request', 'N/A')}ms, "
                    f"Power={r.get('total_power_peak_W', 'N/A')}W"
                )
                lines.append(f"  Lesson: {ep['reflection']}")

        lines.append(
            "\nUse these experiences as reference, but adapt to the current workload's "
            "specific characteristics. Do not blindly copy past configs."
        )
        return "\n".join(lines)

    # ── Utilities ──

    def num_episodes(self) -> int:
        return len(self._load_episodes())

    def clear(self) -> None:
        """Clear all memory (for testing)."""
        if self._episodes_path.exists():
            self._episodes_path.unlink()
        if self._semantics_path.exists():
            self._semantics_path.unlink()
