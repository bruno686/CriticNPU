"""Roofline-based World Model for NPU design agents.

Provides physics-based predictions of how chip parameter changes affect
performance, enabling the LLM agent to reason about design decisions
WITHOUT running the full NeuSim simulator.

Three layers:
  1. Roofline Model: compute ceiling, BW ceiling, ridge point, operator mapping
  2. Power Model: analytical power from chip parameters
  3. Bottleneck Cascade: predict how fixing one bottleneck shifts to the next

No LLM calls in this module — pure deterministic physics.
"""

import json
import math
from dataclasses import dataclass, field
from typing import Any

import sys
from pathlib import Path

_NEUSIM_ROOT = Path(__file__).resolve().parent.parent / "NeuSim"
sys.path.insert(0, str(_NEUSIM_ROOT))

from neusim.configs.chips.ChipConfig import ChipConfig
from neusim_runner import apply_chip_overrides


# ── Operator type classification ────────────────────────────────────────────

def _classify_op(name: str) -> str:
    """Classify an operator name into a type category for learned roofline."""
    n = name.lower()
    if any(k in n for k in ("-q-", "-k-", "-v-", "-qkv-", "proj_q", "proj_k", "proj_v")):
        return "matmul_qkv"
    if any(k in n for k in ("ffn", "gate", "up", "down", "mlp", "fc1", "fc2")):
        return "matmul_ffn"
    if any(k in n for k in ("attention", "attn", "flash")):
        return "attention"
    if any(k in n for k in ("output", "dense", "linear", "proj", "matmul", "einsum")):
        return "matmul_other"
    return "other"


# ── Learned Roofline ────────────────────────────────────────────────────────

class LearnedRoofline:
    """Online-learned efficiency corrections for the roofline model.

    Tracks per-operator-type efficiency factors (η) that capture the gap
    between theoretical roofline ceilings and actual NeuSIM performance.

    Three capabilities:
      1. Learn η online from simulation feedback (EMA with variance)
      2. Surrogate prediction: predict TTFT/TPOT for new configs WITHOUT simulation
      3. Confidence gating: decide whether a simulation call is necessary

    η_compute: ratio of actual SA throughput to theoretical peak
    η_memory:  ratio of actual HBM bandwidth to theoretical peak
    """

    # Minimum simulation rounds before surrogate predictions are trusted
    MIN_UPDATES_FOR_GATING = 3

    def __init__(self, ema_alpha: float = 0.3):
        self.ema_alpha = ema_alpha

        # Per-op-type: η mean and variance (Welford online)
        self._eta_compute: dict[str, float] = {}
        self._eta_memory: dict[str, float] = {}
        self._eta_compute_var: dict[str, float] = {}
        self._eta_memory_var: dict[str, float] = {}
        self._eta_compute_n: dict[str, int] = {}
        self._eta_memory_n: dict[str, int] = {}

        self.n_updates: int = 0

        # Prediction accuracy tracking: list of (predicted, actual) pairs
        self._prediction_errors: list[float] = []  # relative errors

        # Reference operator profile (stored from simulations for surrogate prediction)
        self._ref_operators: list["RooflineOperator"] = []
        self._ref_roofline: "RooflineModel | None" = None

    def update(self, roofline: "RooflineModel") -> None:
        """Update η from a populated RooflineModel (call after populate_from_ops)."""
        by_type: dict[str, list[RooflineOperator]] = {}
        for op in roofline.operators:
            op_type = _classify_op(op.name)
            by_type.setdefault(op_type, []).append(op)

        for op_type, ops in by_type.items():
            compute_etas: list[float] = []
            memory_etas: list[float] = []

            for op in ops:
                if op.sa_time_ns > 0 and op.flop_count > 0 and roofline.sa_compute_gflops > 0:
                    eta_c = (op.flop_count / op.sa_time_ns) / roofline.sa_compute_gflops
                    compute_etas.append(max(0.01, min(eta_c, 1.0)))

                if op.memory_time_ns > 0 and op.memory_traffic_bytes > 0 and roofline.bw_ceiling_gbps > 0:
                    eta_m = (op.memory_traffic_bytes / op.memory_time_ns) / roofline.bw_ceiling_gbps
                    memory_etas.append(max(0.01, min(eta_m, 1.0)))

            # Update η with EMA, track variance with Welford
            if compute_etas:
                obs = sum(compute_etas) / len(compute_etas)
                self._update_eta(op_type, obs, "compute")
            if memory_etas:
                obs = sum(memory_etas) / len(memory_etas)
                self._update_eta(op_type, obs, "memory")

        # Store reference for surrogate prediction
        self._ref_operators = list(roofline.operators)
        self._ref_roofline = roofline
        self.n_updates += 1

    def _update_eta(self, op_type: str, obs: float, kind: str) -> None:
        """Update η mean and variance using EMA + Welford."""
        if kind == "compute":
            eta_dict, var_dict, n_dict = self._eta_compute, self._eta_compute_var, self._eta_compute_n
        else:
            eta_dict, var_dict, n_dict = self._eta_memory, self._eta_memory_var, self._eta_memory_n

        if op_type not in eta_dict:
            eta_dict[op_type] = obs
            var_dict[op_type] = 0.0
            n_dict[op_type] = 1
        else:
            old_mean = eta_dict[op_type]
            eta_dict[op_type] = (1 - self.ema_alpha) * old_mean + self.ema_alpha * obs
            # Welford online variance (biased towards recent)
            n_dict[op_type] += 1
            delta = obs - old_mean
            var_dict[op_type] = (1 - self.ema_alpha) * var_dict[op_type] + self.ema_alpha * delta * delta

    def eta_compute(self, op_type: str) -> float:
        return self._eta_compute.get(op_type, 1.0)

    def eta_memory(self, op_type: str) -> float:
        return self._eta_memory.get(op_type, 1.0)

    def eta_compute_std(self, op_type: str) -> float:
        return math.sqrt(self._eta_compute_var.get(op_type, 0.0))

    def eta_memory_std(self, op_type: str) -> float:
        return math.sqrt(self._eta_memory_var.get(op_type, 0.0))

    def effective_compute(self, op_type: str, theoretical_gflops: float) -> float:
        return self.eta_compute(op_type) * theoretical_gflops

    def effective_bw(self, op_type: str, theoretical_gbps: float) -> float:
        return self.eta_memory(op_type) * theoretical_gbps

    def effective_ridge(self, op_type: str, theoretical_gflops: float, theoretical_gbps: float) -> float:
        eff_bw = self.effective_bw(op_type, theoretical_gbps)
        return self.effective_compute(op_type, theoretical_gflops) / eff_bw if eff_bw > 0 else float("inf")

    # ── Surrogate Prediction ──────────────────────────────────────────────

    def surrogate_predict(
        self,
        new_overrides: dict[str, Any],
        base_config: dict[str, Any],
        upper_bounds: dict[str, float] | None = None,
    ) -> dict[str, float] | None:
        """Predict TTFT/TPOT for a new config WITHOUT running NeuSIM.

        Uses the stored operator profile from the last simulation and
        scales each operator's time by the ratio of new/old effective ceilings.

        Returns dict with predicted_TTFT_sec, predicted_TPOT_ms, confidence,
        or None if insufficient data.
        """
        if not self._ref_operators or self._ref_roofline is None or self.n_updates < 1:
            return None

        ref = self._ref_roofline

        # Build new roofline for proposed config
        merged = apply_chip_overrides(dict(base_config), new_overrides, upper_bounds=upper_bounds)
        new_chip = ChipConfig.model_validate(merged)
        new_sa_compute = new_chip.peak_SA_tflops_per_sec * 1e3  # GFLOPS
        new_bw = new_chip.hbm_bw_GBps
        new_ici = new_chip.ici_bw_GBps

        prefill_time_ns = 0.0
        decode_time_ns = 0.0

        for op in self._ref_operators:
            if op.execution_time_ns == 0:
                continue

            op_type = _classify_op(op.name)

            # Effective ceilings for old and new configs
            eff_old_c = self.effective_compute(op_type, ref.sa_compute_gflops)
            eff_new_c = self.effective_compute(op_type, new_sa_compute)
            eff_old_m = self.effective_bw(op_type, ref.bw_ceiling_gbps)
            eff_new_m = self.effective_bw(op_type, new_bw)

            # Scale time based on regime
            if op.regime == "ici-bound":
                ratio = ref.ici_bw_gbps / new_ici if new_ici > 0 else 1.0
            elif op.regime == "compute-bound":
                ratio = eff_old_c / eff_new_c if eff_new_c > 0 else 1.0
            else:  # memory-bound
                ratio = eff_old_m / eff_new_m if eff_new_m > 0 else 1.0

            new_time = op.execution_time_ns * ratio

            if op.phase == "prefill":
                prefill_time_ns += new_time
            else:
                decode_time_ns += new_time

        # Convert to standard units (approximate — ignores PP/TP scaling factors
        # but captures relative changes accurately)
        ref_prefill_ns = sum(o.execution_time_ns for o in self._ref_operators if o.phase == "prefill")
        ref_decode_ns = sum(o.execution_time_ns for o in self._ref_operators if o.phase == "decode")

        # Use ratios relative to reference to predict absolute values
        ref_ttft = getattr(ref, '_ref_ttft', None)
        ref_tpot = getattr(ref, '_ref_tpot', None)

        if ref_ttft and ref_prefill_ns > 0:
            predicted_ttft = ref_ttft * (prefill_time_ns / ref_prefill_ns)
        else:
            predicted_ttft = prefill_time_ns / 1e9  # fallback

        if ref_tpot and ref_decode_ns > 0:
            predicted_tpot = ref_tpot * (decode_time_ns / ref_decode_ns)
        else:
            predicted_tpot = decode_time_ns / 1e6  # fallback

        return {
            "predicted_TTFT_sec": predicted_ttft,
            "predicted_TPOT_ms": predicted_tpot,
            "confidence": self.confidence_score(),
        }

    def record_prediction_error(self, predicted: float, actual: float) -> None:
        """Record a prediction vs actual result for confidence calibration."""
        if actual > 0:
            rel_error = abs(predicted - actual) / actual
            self._prediction_errors.append(rel_error)
            # Keep last 20
            if len(self._prediction_errors) > 20:
                self._prediction_errors = self._prediction_errors[-20:]

    def confidence_score(self) -> float:
        """Confidence in surrogate predictions. 0.0 = no confidence, 1.0 = perfect.

        Based on: number of updates and recent prediction accuracy (MAPE).
        η variance is NOT used — it reflects real config-dependence, not uncertainty.
        """
        if self.n_updates < self.MIN_UPDATES_FOR_GATING:
            return 0.0

        # Factor 1: enough updates (ramps from 0 to 1 over MIN..MIN+3)
        update_factor = min(1.0, (self.n_updates - self.MIN_UPDATES_FOR_GATING + 1) / 3)

        # Factor 2: recent prediction accuracy (the only real measure of trustworthiness)
        if len(self._prediction_errors) >= 2:
            recent = self._prediction_errors[-10:]
            mean_error = sum(recent) / len(recent)
            # MAPE < 5% → high confidence, MAPE > 30% → no confidence
            accuracy_factor = max(0.0, min(1.0, (0.30 - mean_error) / 0.25))
        else:
            accuracy_factor = 0.5  # not enough data, moderate default

        return update_factor * accuracy_factor

    def should_simulate(
        self,
        predicted_metric: float,
        best_metric: float,
        minimize: bool = True,
    ) -> bool:
        """Decide whether to run NeuSIM or trust the surrogate.

        Returns True if simulation is recommended, False if surrogate
        predicts this config is not worth simulating.
        """
        conf = self.confidence_score()

        # Not enough confidence yet → always simulate
        if conf < 0.3:
            return True

        import os
        _margin_base = float(os.environ.get("NPUAGENT_GATING_MARGIN", "0.05"))
        margin = _margin_base * (1 - conf)  # tighter margin at higher confidence
        if minimize:
            if predicted_metric > best_metric * (1 + margin):
                return False
        else:
            if predicted_metric < best_metric * (1 - margin):
                return False

        return True

    def store_reference(self, ttft: float, tpot: float) -> None:
        """Store reference TTFT/TPOT from the most recent simulation.

        Needed for absolute surrogate predictions.
        """
        if self._ref_roofline is not None:
            self._ref_roofline._ref_ttft = ttft
            self._ref_roofline._ref_tpot = tpot

    # ── Formatting ────────────────────────────────────────────────────────

    def format_for_llm(self) -> str:
        """Format learned corrections as structured text for the LLM."""
        if self.n_updates == 0:
            return ""

        lines = [f"\n### Learned Efficiency Corrections (from {self.n_updates} simulation{'s' if self.n_updates > 1 else ''}, confidence={self.confidence_score():.0%})"]
        lines.append("Theoretical roofline assumes 100% utilization. Actual efficiency per operator type:")
        lines.append(f"{'Op Type':<16} {'η_compute':>10} {'±σ':>6} {'η_memory':>10} {'±σ':>6} {'Meaning':>30}")
        lines.append("-" * 84)

        all_types = sorted(set(list(self._eta_compute.keys()) + list(self._eta_memory.keys())))
        for op_type in all_types:
            eta_c = self._eta_compute.get(op_type, 1.0)
            eta_m = self._eta_memory.get(op_type, 1.0)
            std_c = self.eta_compute_std(op_type)
            std_m = self.eta_memory_std(op_type)
            parts = []
            if eta_c < 0.9:
                parts.append(f"SA util {eta_c:.0%}")
            if eta_m < 0.9:
                parts.append(f"BW util {eta_m:.0%}")
            meaning = ", ".join(parts) if parts else "near theoretical"
            lines.append(f"{op_type:<16} {eta_c:>10.2f} {std_c:>5.3f} {eta_m:>10.2f} {std_m:>5.3f} {meaning:>30}")

        lines.append("")
        lines.append("Use EFFECTIVE ceilings (= theoretical × η) for more accurate predictions.")
        return "\n".join(lines)


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class RooflineOperator:
    """A single operator mapped onto the roofline plane."""
    name: str
    phase: str                          # "prefill" or "decode"
    flop_count: int                     # total FLOPs (count * per-invocation)
    memory_traffic_bytes: int           # total memory traffic
    arithmetic_intensity: float         # flop_count / memory_traffic_bytes
    count: int
    bounded_by: str                     # from NeuSim: "Compute", "Memory", "ICI/NVLink"

    # Timing from simulation (ns, weighted by count)
    execution_time_ns: int = 0
    sa_time_ns: int = 0
    vu_time_ns: int = 0
    memory_time_ns: int = 0
    ici_time_ns: int = 0

    # Utilization
    flops_util: float = 0.0
    hbm_bw_util: float = 0.0

    # Roofline classification
    regime: str = ""                    # "compute-bound", "memory-bound", "ici-bound"
    distance_to_ridge: float = 0.0     # log2(intensity / ridge_point), positive = compute side


@dataclass
class RooflineModel:
    """Complete roofline representation for one chip configuration."""
    # Chip ceilings
    sa_compute_gflops: float            # num_sa * sa_dim^2 * 2 * freq_GHz
    vu_compute_gflops: float            # num_vu * 8 * 128 * freq_GHz
    total_compute_gflops: float         # sa + vu
    bw_ceiling_gbps: float              # hbm_bw_GBps
    ridge_point: float                  # sa_compute_gflops / bw_ceiling_gbps (FLOP/byte)
    ici_bw_gbps: float

    # Chip params (for reference)
    num_sa: int = 0
    num_vu: int = 0
    sa_dim: int = 0
    freq_ghz: float = 0.0
    vmem_size_mb: int = 0

    # Power
    total_power_peak_W: float = 0.0
    power_budget_W: float = 0.0
    power_headroom_W: float = 0.0

    # Operators
    operators: list[RooflineOperator] = field(default_factory=list)

    # Phase-level bottleneck distribution (by weighted time)
    prefill_pct_compute: float = 0.0
    prefill_pct_memory: float = 0.0
    prefill_pct_ici: float = 0.0
    decode_pct_compute: float = 0.0
    decode_pct_memory: float = 0.0
    decode_pct_ici: float = 0.0

    prefill_dominant: str = ""          # "compute", "memory", "ici"
    decode_dominant: str = ""

    # Aggregate timing
    prefill_total_time_ns: int = 0
    decode_total_time_ns: int = 0


@dataclass
class CascadePrediction:
    """Prediction of how parameter changes cascade through bottlenecks."""
    param_changes: dict[str, Any]

    old_sa_compute: float
    new_sa_compute: float
    old_bw: float
    new_bw: float
    old_ridge: float
    new_ridge: float

    new_power_W: float
    power_valid: bool

    # Operator regime migrations
    ops_becoming_compute_bound: list[str] = field(default_factory=list)
    ops_becoming_memory_bound: list[str] = field(default_factory=list)

    # Estimated speedups (>1 means faster)
    estimated_prefill_speedup: float = 1.0
    estimated_decode_speedup: float = 1.0

    rationale: str = ""


# ── Roofline Construction ────────────────────────────────────────────────────

def build_roofline(
    chip_overrides: dict[str, Any],
    base_config: dict[str, Any],
    power_budget_W: float,
    upper_bounds: dict[str, float] | None = None,
) -> RooflineModel:
    """Build a roofline model from chip parameters (no simulation needed)."""
    merged = dict(base_config)
    merged = apply_chip_overrides(merged, chip_overrides, upper_bounds=upper_bounds)

    chip_cfg = ChipConfig.model_validate(merged)

    sa_compute = chip_cfg.peak_SA_tflops_per_sec * 1e3   # GFLOPS
    vu_compute = chip_cfg.peak_VU_tflops_per_sec * 1e3   # GFLOPS
    bw = chip_cfg.hbm_bw_GBps                             # GB/s

    ridge = sa_compute / bw if bw > 0 else float("inf")   # FLOP/byte (SA-based ridge)

    return RooflineModel(
        sa_compute_gflops=sa_compute,
        vu_compute_gflops=vu_compute,
        total_compute_gflops=sa_compute + vu_compute,
        bw_ceiling_gbps=bw,
        ridge_point=ridge,
        ici_bw_gbps=chip_cfg.ici_bw_GBps,
        num_sa=chip_cfg.num_sa,
        num_vu=chip_cfg.num_vu,
        sa_dim=chip_cfg.sa_dim,
        freq_ghz=chip_cfg.freq_GHz,
        vmem_size_mb=chip_cfg.vmem_size_MB,
        total_power_peak_W=chip_cfg.total_power_peak_W,
        power_budget_W=power_budget_W,
        power_headroom_W=power_budget_W - chip_cfg.total_power_peak_W,
    )


def populate_from_ops(
    model: RooflineModel,
    prefill_ops: list,
    decode_ops: list,
    learned: LearnedRoofline | None = None,
) -> RooflineModel:
    """Map NeuSim operators onto the roofline plane.

    Args:
        prefill_ops, decode_ops: lists of Operator objects from NeuSim.
        learned: optional learned efficiency corrections; if provided,
                 regime classification uses per-op-type effective ridge points.
    """
    model.operators = []

    for phase, ops in [("prefill", prefill_ops), ("decode", decode_ops)]:
        for op in ops:
            s = op.stats
            weighted_time = s.execution_time_ns * s.count
            flops = s.flop_count * s.count
            mem_bytes = s.memory_traffic_bytes * s.count

            intensity = flops / mem_bytes if mem_bytes > 0 else float("inf")

            # Use effective ridge point if learned corrections available
            op_type = _classify_op(op.description or op.name)
            if learned and learned.n_updates > 0:
                ridge = learned.effective_ridge(
                    op_type, model.sa_compute_gflops, model.bw_ceiling_gbps
                )
            else:
                ridge = model.ridge_point

            # Classify regime
            if s.bounded_by == "ICI/NVLink" or s.ici_time_ns > max(s.sa_time_ns, s.vu_time_ns, s.memory_time_ns):
                regime = "ici-bound"
            elif intensity > ridge and ridge > 0:
                regime = "compute-bound"
            else:
                regime = "memory-bound"

            dist = math.log2(intensity / ridge) if ridge > 0 and intensity > 0 else 0.0

            ro = RooflineOperator(
                name=op.description or op.name,
                phase=phase,
                flop_count=flops,
                memory_traffic_bytes=mem_bytes,
                arithmetic_intensity=intensity,
                count=s.count,
                bounded_by=s.bounded_by,
                execution_time_ns=weighted_time,
                sa_time_ns=s.sa_time_ns * s.count,
                vu_time_ns=s.vu_time_ns * s.count,
                memory_time_ns=s.memory_time_ns * s.count,
                ici_time_ns=s.ici_time_ns * s.count,
                flops_util=s.flops_util,
                hbm_bw_util=s.hbm_bw_util,
                regime=regime,
                distance_to_ridge=dist,
            )
            model.operators.append(ro)

    # Aggregate phase-level stats
    _aggregate_phase_stats(model)
    return model


def _aggregate_phase_stats(model: RooflineModel) -> None:
    """Compute phase-level bottleneck percentages from operators."""
    for phase in ["prefill", "decode"]:
        ops = [o for o in model.operators if o.phase == phase]
        total_time = sum(o.execution_time_ns for o in ops)

        if total_time == 0:
            continue

        compute_time = sum(o.execution_time_ns for o in ops if o.regime == "compute-bound")
        memory_time = sum(o.execution_time_ns for o in ops if o.regime == "memory-bound")
        ici_time = sum(o.execution_time_ns for o in ops if o.regime == "ici-bound")

        pct_c = compute_time / total_time * 100
        pct_m = memory_time / total_time * 100
        pct_i = ici_time / total_time * 100

        dominant = "compute" if pct_c >= pct_m and pct_c >= pct_i else (
            "memory" if pct_m >= pct_i else "ici"
        )

        if phase == "prefill":
            model.prefill_pct_compute = pct_c
            model.prefill_pct_memory = pct_m
            model.prefill_pct_ici = pct_i
            model.prefill_dominant = dominant
            model.prefill_total_time_ns = total_time
        else:
            model.decode_pct_compute = pct_c
            model.decode_pct_memory = pct_m
            model.decode_pct_ici = pct_i
            model.decode_dominant = dominant
            model.decode_total_time_ns = total_time


# ── Power Prediction ─────────────────────────────────────────────────────────

def predict_power(
    base_config: dict[str, Any],
    overrides: dict[str, Any],
    upper_bounds: dict[str, float] | None = None,
) -> float:
    """Predict total_power_peak_W without running simulation."""
    merged = dict(base_config)
    merged = apply_chip_overrides(merged, overrides, upper_bounds=upper_bounds)
    chip_cfg = ChipConfig.model_validate(merged)
    return chip_cfg.total_power_peak_W


# ── Cascade Prediction ───────────────────────────────────────────────────────

def predict_cascade(
    model: RooflineModel,
    param_changes: dict[str, Any],
    base_config: dict[str, Any],
    upper_bounds: dict[str, float] | None = None,
    learned: LearnedRoofline | None = None,
) -> CascadePrediction:
    """Predict how parameter changes cascade through bottlenecks.

    Uses the roofline model to estimate latency impact without simulation.
    If a LearnedRoofline is provided, uses per-op-type effective ceilings
    for more accurate predictions.
    """
    # Build new roofline from proposed changes
    merged = dict(base_config)
    current_overrides = {
        "num_sa": model.num_sa, "num_vu": model.num_vu,
        "sa_dim": model.sa_dim, "freq_GHz": model.freq_ghz,
        "vmem_size_MB": model.vmem_size_mb,
    }
    new_overrides = {**current_overrides, **param_changes}

    # Compute new ceilings
    new_merged = apply_chip_overrides(dict(merged), new_overrides, upper_bounds=upper_bounds)
    new_chip = ChipConfig.model_validate(new_merged)

    new_sa_compute = new_chip.peak_SA_tflops_per_sec * 1e3  # GFLOPS
    new_bw = new_chip.hbm_bw_GBps
    new_ridge = new_sa_compute / new_bw if new_bw > 0 else float("inf")
    new_power = new_chip.total_power_peak_W

    use_learned = learned is not None and learned.n_updates > 0

    # Predict per-operator latency changes
    ops_to_compute = []
    ops_to_memory = []
    prefill_old_time = 0
    prefill_new_time = 0
    decode_old_time = 0
    decode_new_time = 0

    for op in model.operators:
        old_time = op.execution_time_ns
        if old_time == 0:
            continue

        op_type = _classify_op(op.name)

        # Effective ceilings for this operator type
        if use_learned:
            eff_old_compute = learned.effective_compute(op_type, model.sa_compute_gflops)
            eff_new_compute = learned.effective_compute(op_type, new_sa_compute)
            eff_old_bw = learned.effective_bw(op_type, model.bw_ceiling_gbps)
            eff_new_bw = learned.effective_bw(op_type, new_bw)
            eff_new_ridge = eff_new_compute / eff_new_bw if eff_new_bw > 0 else float("inf")
        else:
            eff_old_compute = model.sa_compute_gflops
            eff_new_compute = new_sa_compute
            eff_old_bw = model.bw_ceiling_gbps
            eff_new_bw = new_bw
            eff_new_ridge = new_ridge

        # Determine new regime using effective ridge
        new_regime = op.regime  # default: same
        if op.regime != "ici-bound":
            if op.arithmetic_intensity > eff_new_ridge:
                new_regime = "compute-bound"
            else:
                new_regime = "memory-bound"

        # Track regime migrations
        if op.regime == "memory-bound" and new_regime == "compute-bound":
            ops_to_compute.append(op.name)
        elif op.regime == "compute-bound" and new_regime == "memory-bound":
            ops_to_memory.append(op.name)

        # Estimate new execution time using effective ceilings
        if op.regime == "ici-bound":
            old_ici = model.ici_bw_gbps
            new_ici = new_chip.ici_bw_GBps
            ratio = old_ici / new_ici if new_ici > 0 else 1.0
            new_time = old_time * ratio
        elif op.regime == "compute-bound":
            ratio = eff_old_compute / eff_new_compute if eff_new_compute > 0 else 1.0
            new_time = old_time * ratio
        else:
            ratio = eff_old_bw / eff_new_bw if eff_new_bw > 0 else 1.0
            new_time = old_time * ratio

        if op.phase == "prefill":
            prefill_old_time += old_time
            prefill_new_time += new_time
        else:
            decode_old_time += old_time
            decode_new_time += new_time

    prefill_speedup = prefill_old_time / prefill_new_time if prefill_new_time > 0 else 1.0
    decode_speedup = decode_old_time / decode_new_time if decode_new_time > 0 else 1.0

    return CascadePrediction(
        param_changes=param_changes,
        old_sa_compute=model.sa_compute_gflops,
        new_sa_compute=new_sa_compute,
        old_bw=model.bw_ceiling_gbps,
        new_bw=new_bw,
        old_ridge=model.ridge_point,
        new_ridge=new_ridge,
        new_power_W=new_power,
        power_valid=new_power <= model.power_budget_W,
        ops_becoming_compute_bound=ops_to_compute,
        ops_becoming_memory_bound=ops_to_memory,
        estimated_prefill_speedup=prefill_speedup,
        estimated_decode_speedup=decode_speedup,
    )


# ── Direction Suggestion ─────────────────────────────────────────────────────

def suggest_directions(
    model: RooflineModel,
    base_config: dict[str, Any],
    upper_bounds: dict[str, float] | None = None,
    ttft_weight: float = 0.5,
    tpot_weight: float = 0.5,
    learned: LearnedRoofline | None = None,
) -> list[dict[str, Any]]:
    """Use roofline analysis to suggest promising parameter directions.

    Returns a ranked list of suggestions, each with:
        param_changes, rationale, cascade (CascadePrediction)
    """
    suggestions = []

    # Determine what to optimize based on weights
    optimize_prefill = ttft_weight > 0
    optimize_decode = tpot_weight > 0

    # Generate candidate changes based on dominant bottleneck
    candidates = []

    if optimize_prefill and model.prefill_dominant == "compute":
        if model.power_headroom_W > 30:
            candidates.append({
                "changes": {"num_sa": model.num_sa + 2, "num_vu_ports": model.num_vu},
                "rationale": f"Prefill is {model.prefill_pct_compute:.0f}% compute-bound with {model.power_headroom_W:.0f}W headroom — add SAs",
            })
        # Lower freq to save power, then add more SAs
        if model.freq_ghz > 0.6:
            new_freq = round(model.freq_ghz - 0.2, 2)
            candidates.append({
                "changes": {"freq_GHz": new_freq, "num_sa": model.num_sa + 2, "num_vu_ports": model.num_vu},
                "rationale": f"Prefill compute-bound: lower freq to {new_freq} to free power for +2 SAs",
            })
        if model.sa_dim < 256:
            next_dim = 256 if model.sa_dim == 128 else 128
            candidates.append({
                "changes": {"sa_dim": next_dim},
                "rationale": f"Prefill compute-bound: increase SA dim {model.sa_dim}→{next_dim} (4x compute per SA)",
            })

    if optimize_prefill and model.prefill_dominant == "memory":
        candidates.append({
            "changes": {"vmem_size_MB": min(512, model.vmem_size_mb + 128)},
            "rationale": f"Prefill is {model.prefill_pct_memory:.0f}% memory-bound — increase SRAM to reduce HBM traffic",
        })
        if model.num_sa > 3:
            candidates.append({
                "changes": {"num_sa": model.num_sa - 2, "num_vu_ports": model.num_vu},
                "rationale": f"Prefill memory-bound: reduce SAs to save power (compute is not the bottleneck)",
            })

    if optimize_decode and model.decode_dominant == "memory":
        # Decode is almost always memory-bound; minimize compute to save power
        if model.num_sa > 2:
            candidates.append({
                "changes": {"num_sa": max(2, model.num_sa - 2), "num_vu_ports": model.num_vu},
                "rationale": f"Decode is {model.decode_pct_memory:.0f}% memory-bound — fewer SAs save power without hurting TPOT",
            })
        if model.freq_ghz > 0.5:
            candidates.append({
                "changes": {"freq_GHz": 0.5},
                "rationale": "Decode memory-bound: minimize freq to save power (compute irrelevant)",
            })

    if optimize_prefill and model.prefill_dominant == "ici":
        candidates.append({
            "changes": {"ici_bw_GBps": upper_bounds.get("ici_bw_GBps", 125) if upper_bounds else 125},
            "rationale": f"Prefill is {model.prefill_pct_ici:.0f}% ICI-bound — maximize ICI bandwidth",
        })

    # Evaluate each candidate with cascade prediction
    for c in candidates:
        try:
            cascade = predict_cascade(model, c["changes"], base_config, upper_bounds, learned=learned)
            if not cascade.power_valid:
                c["rationale"] += f" [POWER INVALID: {cascade.new_power_W:.0f}W > {model.power_budget_W:.0f}W]"
            suggestions.append({
                "param_changes": c["changes"],
                "rationale": c["rationale"],
                "cascade": cascade,
                "estimated_prefill_speedup": cascade.estimated_prefill_speedup,
                "estimated_decode_speedup": cascade.estimated_decode_speedup,
                "power_valid": cascade.power_valid,
            })
        except Exception:
            continue

    # Sort: valid configs first, then by estimated improvement
    def _sort_key(s):
        valid = 0 if s["power_valid"] else 1
        speedup = 0.0
        if ttft_weight > 0:
            speedup += ttft_weight * s["estimated_prefill_speedup"]
        if tpot_weight > 0:
            speedup += tpot_weight * s["estimated_decode_speedup"]
        return (valid, -speedup)

    suggestions.sort(key=_sort_key)
    return suggestions


# ── Formatting for LLM Context ───────────────────────────────────────────────

def format_for_llm(
    model: RooflineModel,
    cascade: CascadePrediction | None = None,
    suggestions: list[dict[str, Any]] | None = None,
    learned: LearnedRoofline | None = None,
) -> str:
    """Format roofline analysis as structured text for injection into LLM prompt."""
    lines = ["## Roofline World Model"]

    use_learned = learned is not None and learned.n_updates > 0

    # Chip ceilings
    lines.append(f"- SA Compute ceiling: {model.sa_compute_gflops:.1f} GFLOPS (theoretical) "
                 f"({model.num_sa} SAs × {model.sa_dim}×{model.sa_dim} × 2 × {model.freq_ghz} GHz)")
    lines.append(f"- VU Compute ceiling: {model.vu_compute_gflops:.1f} GFLOPS "
                 f"({model.num_vu} VUs)")
    lines.append(f"- HBM BW ceiling: {model.bw_ceiling_gbps:.1f} GB/s (theoretical)")
    lines.append(f"- Theoretical ridge point: {model.ridge_point:.2f} FLOP/byte")
    lines.append(f"- Power: {model.total_power_peak_W:.1f}W / {model.power_budget_W:.0f}W "
                 f"(headroom: {model.power_headroom_W:.1f}W)")

    # Learned corrections summary
    if use_learned:
        lines.append(learned.format_for_llm())
        lines.append("**Use the EFFECTIVE (learned) ceilings below, not the theoretical ones above.**")

    # Phase summaries
    for phase in ["prefill", "decode"]:
        ops = [o for o in model.operators if o.phase == phase]
        if not ops:
            continue

        pct_c = model.prefill_pct_compute if phase == "prefill" else model.decode_pct_compute
        pct_m = model.prefill_pct_memory if phase == "prefill" else model.decode_pct_memory
        pct_i = model.prefill_pct_ici if phase == "prefill" else model.decode_pct_ici
        dominant = model.prefill_dominant if phase == "prefill" else model.decode_dominant

        lines.append(f"\n### {phase.capitalize()} Phase — {dominant.upper()}-bound "
                     f"(Compute {pct_c:.0f}% | Memory {pct_m:.0f}% | ICI {pct_i:.0f}%)")

        # Top 5 operators by time
        top = sorted(ops, key=lambda o: o.execution_time_ns, reverse=True)[:5]
        total_time = sum(o.execution_time_ns for o in ops)
        for op in top:
            pct = op.execution_time_ns / total_time * 100 if total_time > 0 else 0
            op_type = _classify_op(op.name)
            ridge_label = "ridge"
            if use_learned:
                eff_ridge = learned.effective_ridge(
                    op_type, model.sa_compute_gflops, model.bw_ceiling_gbps
                )
                ridge_label = f"eff. ridge={eff_ridge:.1f}"
            lines.append(
                f"  - {op.name} [{op_type}]: {pct:.1f}% of time, {op.regime}, "
                f"intensity={op.arithmetic_intensity:.1f} FLOP/byte "
                f"({'above' if op.distance_to_ridge > 0 else 'below'} {ridge_label} by "
                f"{abs(op.distance_to_ridge):.1f}x in log2), "
                f"SA util={op.flops_util:.0%}, HBM util={op.hbm_bw_util:.0%}"
            )

    # Cascade prediction
    if cascade:
        lines.append(f"\n### Cascade Prediction for proposed changes: {json.dumps(cascade.param_changes)}")
        lines.append(f"- Compute ceiling: {cascade.old_sa_compute:.1f} → {cascade.new_sa_compute:.1f} GFLOPS")
        lines.append(f"- BW ceiling: {cascade.old_bw:.1f} → {cascade.new_bw:.1f} GB/s")
        lines.append(f"- Ridge point: {cascade.old_ridge:.2f} → {cascade.new_ridge:.2f} FLOP/byte")
        lines.append(f"- Power: {cascade.new_power_W:.1f}W {'✓' if cascade.power_valid else '✗ EXCEEDS BUDGET'}")
        lines.append(f"- Estimated prefill speedup: {cascade.estimated_prefill_speedup:.2f}x")
        lines.append(f"- Estimated decode speedup: {cascade.estimated_decode_speedup:.2f}x")
        if cascade.ops_becoming_compute_bound:
            lines.append(f"- Ops shifting to compute-bound: {', '.join(cascade.ops_becoming_compute_bound[:3])}")
        if cascade.ops_becoming_memory_bound:
            lines.append(f"- Ops shifting to memory-bound: {', '.join(cascade.ops_becoming_memory_bound[:3])}")

    # Suggestions
    if suggestions:
        valid = [s for s in suggestions if s["power_valid"]]
        if valid:
            lines.append("\n### Roofline-Suggested Directions (power-valid)")
            for s in valid[:3]:
                lines.append(
                    f"  - {json.dumps(s['param_changes'])}: "
                    f"prefill {s['estimated_prefill_speedup']:.2f}x, "
                    f"decode {s['estimated_decode_speedup']:.2f}x — "
                    f"{s['rationale']}"
                )

    return "\n".join(lines)
