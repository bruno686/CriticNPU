"""Plot latency and power across design iterations."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def plot_iterations(history_path: str, output_path: str = "iterations_report.png", power_budget: float = 350.0):
    with open(history_path) as f:
        history = json.load(f)

    # Try to read power budget from summary.json in same directory
    summary_path = Path(history_path).parent / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        power_budget = summary.get("power_budget_W", power_budget)
        model_name = summary.get("model_name", "")
        par = summary.get("parallelism", {})
    else:
        model_name = ""
        par = {}

    # Filter out error iterations
    history = [h for h in history if "error" not in h]

    iters = [h["iteration"] for h in history]
    ttft = [h["results"]["TTFT_sec"] for h in history]
    tpot = [h["results"]["TPOT_ms_request"] for h in history]
    peak_pwr = [h["results"]["total_power_peak_W"] for h in history]
    avg_pwr_pre = [h["results"]["avg_power_W_prefill"] for h in history]
    avg_pwr_dec = [h["results"]["avg_power_W_decode"] for h in history]
    valid = [
        not h["results"]["out_of_memory"] and h["results"]["total_power_peak_W"] <= power_budget
        for h in history
    ]

    title_str = f"NPU Design Iterations — {model_name}" if model_name else "NPU Design Iterations"
    if par:
        title_str += f" ({par.get('num_chips','')} chips, TP={par.get('tp','')}, PP={par.get('pp','')}), Budget={power_budget:.0f}W"
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title_str, fontsize=14, fontweight="bold")

    # --- TTFT ---
    ax = axes[0, 0]
    colors = ["tab:green" if v else "tab:red" for v in valid]
    ax.bar(iters, ttft, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("TTFT (sec)")
    ax.set_title("Time To First Token (TTFT)")
    ax.set_xticks(iters)
    # Mark best valid
    valid_ttft = [(i, t) for i, t, v in zip(iters, ttft, valid) if v]
    if valid_ttft:
        best_i, best_t = min(valid_ttft, key=lambda x: x[1])
        ax.annotate(f"Best: {best_t:.4f}s\n(iter {best_i})",
                    xy=(best_i, best_t), xytext=(best_i + 1.5, best_t + 0.03),
                    arrowprops=dict(arrowstyle="->", color="black"),
                    fontsize=9, fontweight="bold")

    # --- TPOT ---
    ax = axes[0, 1]
    ax.bar(iters, tpot, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("TPOT (ms/request)")
    ax.set_title("Time Per Output Token (TPOT)")
    ax.set_xticks(iters)
    valid_tpot = [(i, t) for i, t, v in zip(iters, tpot, valid) if v]
    if valid_tpot:
        best_i, best_t = min(valid_tpot, key=lambda x: x[1])
        ax.annotate(f"Best: {best_t:.4f}ms\n(iter {best_i})",
                    xy=(best_i, best_t), xytext=(best_i + 1.5, best_t + 0.5),
                    arrowprops=dict(arrowstyle="->", color="black"),
                    fontsize=9, fontweight="bold")

    # --- Peak Power ---
    ax = axes[1, 0]
    ax.bar(iters, peak_pwr, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=power_budget, color="red", linestyle="--", linewidth=1.5, label=f"Power Budget ({power_budget:.0f}W)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Peak Power (W)")
    ax.set_title("Chip Peak Power (total_power_peak_W)")
    ax.set_xticks(iters)
    ax.legend(fontsize=9)

    # --- Avg Power (prefill & decode) ---
    ax = axes[1, 1]
    x_offset = 0.2
    bars1 = ax.bar([i - x_offset for i in iters], avg_pwr_pre, width=0.4,
                   color="tab:blue", edgecolor="black", linewidth=0.5, label="Avg Power (prefill)")
    bars2 = ax.bar([i + x_offset for i in iters], avg_pwr_dec, width=0.4,
                   color="tab:orange", edgecolor="black", linewidth=0.5, label="Avg Power (decode)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Power (W)")
    ax.set_title("Average Operator Power")
    ax.set_xticks(iters)
    ax.legend(fontsize=9)

    # Add a legend for valid/invalid
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="tab:green", edgecolor="black", label="Valid"),
                       Patch(facecolor="tab:red", edgecolor="black", label="Invalid")]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")

    # Build effective (accumulated) config per iteration
    CONFIG_KEYS = ["num_sa", "num_vu", "sa_dim", "freq_GHz", "vmem_size_MB",
                   "hbm_bw_GBps", "hbm_size_GB", "ici_bw_GBps"]
    effective = {}
    effective_configs = []
    for h in history:
        effective.update(h["chip_overrides"])
        effective_configs.append(dict(effective))

    # Print summary table with effective config
    header = (f"{'Iter':>4}  {'TTFT(s)':>8}  {'TPOT(ms)':>9}  {'PeakPwr(W)':>11}  "
              f"{'AvgPwr_pre':>10}  {'Valid':>5}  │  "
              + "  ".join(f"{k:>12}" for k in CONFIG_KEYS))
    print(f"\n{header}")
    print("-" * len(header))
    for h, v, ecfg in zip(history, valid, effective_configs):
        r = h["results"]
        cfg_vals = "  ".join(f"{str(ecfg.get(k, '-')):>12}" for k in CONFIG_KEYS)
        print(
            f"{h['iteration']:>4}  "
            f"{r['TTFT_sec']:>8.4f}  "
            f"{r['TPOT_ms_request']:>9.4f}  "
            f"{r['total_power_peak_W']:>11.1f}  "
            f"{r['avg_power_W_prefill']:>10.2f}  "
            f"{'Y' if v else 'N':>5}  │  "
            f"{cfg_vals}"
        )


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "design_results/llama3-8b_20260309_170210/history.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "design_results/llama3-8b_20260309_170210/iterations_report.png"
    plot_iterations(path, out)
