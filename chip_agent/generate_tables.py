"""Generate 17 LaTeX tables from experiment_480 results.

Table 0 (main): Win rate summary across 16 configs per model
Tables 1-16: Per-config detailed results (all 6 models × 8 methods)

Output: /mnt/data1/yh84/chip/current_result.tex
"""

import json
import os
import re
import sys
from pathlib import Path

LOGDIR = Path("experiment_480/logs")
RESDIR = Path("experiment_480/results")

MODELS_DISPLAY = [
    ("llama3-8b", "Llama3-8B"),
    ("llama2-13b", "Llama2-13B"),
    ("llama3-70b", "Llama3-70B"),
    ("llama3_1-405b", "Llama3.1-405B"),
    ("deepseekv2-236b", "DSv2-236B"),
    ("deepseekv3-671b", "DSv3-671B"),
]

METHODS = ["BO", "CMA-ES", "Greedy", "Grid", "ChatDSE", "LLM-DSE", "LLM-Box", "NPUAgent"]

# Config descriptions for captions
CONFIGS_DESC = {
    "C1":  {"small": "TP=2, PP=4, BS=16, 250W", "large": "TP=4, PP=16, BS=16, 250W", "dsv2": "TP=2, PP=8, BS=16, 250W"},
    "C2":  {"small": "TP=2, PP=4, BS=16, 300W", "large": "TP=4, PP=16, BS=16, 300W", "dsv2": "TP=2, PP=8, BS=16, 300W"},
    "C3":  {"small": "TP=2, PP=4, BS=16, 400W", "large": "TP=4, PP=16, BS=16, 400W", "dsv2": "TP=2, PP=8, BS=16, 400W"},
    "C4":  {"small": "TP=4, PP=2, BS=16, 250W", "large": "TP=8, PP=8, BS=16, 250W", "dsv2": "TP=4, PP=4, BS=16, 250W"},
    "C5":  {"small": "TP=4, PP=2, BS=16, 300W", "large": "TP=8, PP=8, BS=16, 300W", "dsv2": "TP=4, PP=4, BS=16, 300W"},
    "C6":  {"small": "TP=4, PP=2, BS=16, 400W", "large": "TP=8, PP=8, BS=16, 400W", "dsv2": "TP=4, PP=4, BS=16, 400W"},
    "C7":  {"small": "TP=8, PP=1, BS=16, 250W", "large": "TP=16, PP=4, BS=16, 250W", "dsv2": "TP=8, PP=2, BS=16, 250W"},
    "C8":  {"small": "TP=8, PP=1, BS=16, 300W", "large": "TP=16, PP=4, BS=16, 300W", "dsv2": "TP=8, PP=2, BS=16, 300W"},
    "C9":  {"small": "TP=8, PP=1, BS=16, 400W", "large": "TP=16, PP=4, BS=16, 400W", "dsv2": "TP=8, PP=2, BS=16, 400W"},
    "C10": {"small": "TP=4, PP=2, BS=1, 300W",  "large": "TP=8, PP=8, BS=1, 300W",  "dsv2": "TP=4, PP=4, BS=1, 300W"},
    "C11": {"small": "TP=4, PP=2, BS=32, 300W", "large": "TP=8, PP=8, BS=32, 300W", "dsv2": "TP=4, PP=4, BS=32, 300W"},
    "C12": {"small": "TP=4, PP=2, BS=48, 300W", "large": "TP=8, PP=8, BS=48, 300W", "dsv2": "TP=4, PP=4, BS=48, 300W"},
    "C13": {"small": "TP=2, PP=4, BS=1, 300W",  "large": "TP=4, PP=16, BS=1, 300W", "dsv2": "TP=2, PP=8, BS=1, 300W"},
    "C14": {"small": "TP=8, PP=1, BS=1, 300W",  "large": "TP=16, PP=4, BS=1, 300W", "dsv2": "TP=8, PP=2, BS=1, 300W"},
    "C15": {"small": "TP=4, PP=2, BS=16, 350W", "large": "TP=8, PP=8, BS=16, 350W", "dsv2": "TP=4, PP=4, BS=16, 350W"},
    "C16": {"small": "TP=2, PP=4, BS=16, 350W", "large": "TP=4, PP=16, BS=16, 350W","dsv2": "TP=2, PP=8, BS=16, 350W"},
}


MIN_FREQ_GHZ = 0.5  # hard lower bound; below this simulator numerical artifacts kick in


def _parse_llm_best_from_iters(text: str) -> tuple[float | None, float | None]:
    """For LLM-DSE / LLM-Box: re-derive best TTFT/TPOT filtering freq<MIN_FREQ (exploit).

    These baselines accept LLM-proposed freq_GHz without clamping, so LLM can
    return freq=0.005 which produces unrealistic TTFT via simulator division.
    NPUAgent has MIN_FREQ_GHZ clamp, so fairness requires the same filter here.

    Scans each "[iter NN] ... f=X.XX ... TTFT=... TPOT=... ... VALID" line,
    keeps only rows where f >= 0.5 AND status is VALID. Splits prefill/decode
    by the ">>> *PREFILL optimization <<<" / ">>> *DECODE optimization <<<"
    markers; returns (best_TTFT_in_prefill, best_TPOT_in_decode).
    """
    # Split by phase markers
    pre_start = re.search(r">>>[^<]*PREFILL optimization", text)
    dec_start = re.search(r">>>[^<]*DECODE optimization", text)
    if pre_start and dec_start:
        prefill_text = text[pre_start.start():dec_start.start()]
        decode_text = text[dec_start.start():]
    else:
        # Non-disaggregated: single phase, cannot split cleanly
        prefill_text = text
        decode_text = text

    iter_re = re.compile(
        r"\[iter \d+\]\s+num_sa=\s*\d+\s+num_vu=\s*\d+\s+sa=\s*\d+\s+f=([0-9.]+)\s+vmem=\s*\d+\s+.*?"
        r"TTFT=([0-9.]+)s\s+TPOT=([0-9.]+)ms\s+Pwr=\d+W\s+(\w+)"
    )

    def min_valid(phase_text, metric_idx):
        best = None
        for m in iter_re.finditer(phase_text):
            freq = float(m.group(1))
            status = m.group(4)
            if freq < MIN_FREQ_GHZ:
                continue
            if status != "VALID":
                continue
            val = float(m.group(2)) if metric_idx == "ttft" else float(m.group(3))
            if best is None or val < best:
                best = val
        return best

    ttft = min_valid(prefill_text, "ttft")
    tpot = min_valid(decode_text, "tpot")
    return ttft, tpot


def parse_log(logfile):
    """Extract TTFT, TPOT, sims from a log file."""
    text = Path(logfile).read_text()
    name = Path(logfile).stem
    ttft = tpot = sims = None

    # For LLM-DSE / LLM-Box: re-derive best from per-iter lines, filtering the
    # freq<MIN_FREQ_GHZ exploit (baselines lack the clamp NPUAgent has).
    if "_llm_dse_" in name or "_llm_box_" in name:
        ttft, tpot = _parse_llm_best_from_iters(text)

    # Try Combined: line first (the FINAL result) — for methods without the exploit
    if ttft is None or tpot is None:
        m = re.search(r"Combined:.*?TTFT=([0-9.]+)s.*?TPOT=([0-9.]+)ms", text)
        if m:
            if ttft is None: ttft = float(m.group(1))
            if tpot is None: tpot = float(m.group(2))

    # Try separate Prefill/Decode best result lines
    if ttft is None:
        m_p = re.search(r"(?:Prefill|PREFILL).*?TTFT=([0-9.]+)", text)
        if m_p:
            ttft = float(m_p.group(1))
    if tpot is None:
        m_d = re.search(r"(?:Decode|DECODE).*?TPOT=([0-9.]+)", text)
        if m_d:
            tpot = float(m_d.group(1))

    # Last resort: find the LAST occurrence of TTFT= / TPOT=
    # Skip this fallback for LLM-DSE/LLM-Box (their last line is the exploit result)
    if ttft is None and "_llm_dse_" not in name and "_llm_box_" not in name:
        all_ttft = re.findall(r"TTFT=([0-9.]+)s", text)
        if all_ttft:
            ttft = float(all_ttft[-1])
    if tpot is None and "_llm_dse_" not in name and "_llm_box_" not in name:
        all_tpot = re.findall(r"TPOT=([0-9.]+)ms", text)
        if all_tpot:
            tpot = float(all_tpot[-1])

    # Sims count — REAL simulator calls (not iteration budget).
    #   ChatDSE          : ">>> Verifying with NeuSim" per phase
    #   LLM-DSE          : "[iter NN]" lines minus DUP minus ERROR (both = no sim)
    #   LLM-Box          : "[iter NN]" lines minus "critic REJECTED" minus DUP minus ERROR
    #   NPUAgent         : 9 seeds × n_phases + iter sims + refine sims
    #   BO/CMA-ES/Greedy : sum of "Total simulations: N" (once per phase)
    #   Grid             : product of iter count OR fall back to 720 (16-d grid)
    if "_chatdse_" in name:
        sims = text.count(">>> Verifying with NeuSim")
        if sims == 0 and ttft is not None:
            sims = 1
    elif "_llm_dse_" in name:
        iter_lines = re.findall(r"^  \[iter \d+\].*$", text, flags=re.MULTILINE)
        dup_or_err = sum(1 for ln in iter_lines if "DUP" in ln or "ERROR" in ln)
        sims = len(iter_lines) - dup_or_err
    elif "_llm_box_" in name:
        iter_lines = re.findall(r"^  \[iter \d+\].*$", text, flags=re.MULTILINE)
        reject_or_dup_or_err = sum(1 for ln in iter_lines
                                    if "critic REJECTED" in ln or "DUP" in ln or "ERROR" in ln)
        sims = len(iter_lines) - reject_or_dup_or_err
    elif "_agent_" in name:
        # Count actual simulator invocations:
        # (1) seed phase: 9 seeds per phase, N phases = count of "--- Seed phase done"
        # (2) LLM loop: each printed "=== NeuSIM Simulation Results ===" banner = one sim
        # (3) local refinement: "Local refinement total: N simulations across ..." per phase
        n_phases = text.count("--- Seed phase done")
        if n_phases == 0:  # fallback for older logs
            n_phases = 2 if ("DISAGGREGATED" in text or "PREFILL" in text.upper() and "DECODE" in text.upper()) else 1
        seed_sims = 9 * n_phases
        iter_sims = text.count("=== NeuSIM Simulation Results ===")
        refine_sims = sum(int(m) for m in re.findall(r"Local refinement total:\s*(\d+)\s*simulations", text))
        sims = seed_sims + iter_sims + refine_sims
    else:
        # Classical baselines: sum actual sim counts across both phases.
        totals = re.findall(r"Total simulations:\s*(\d+)", text)
        if totals:
            sims = sum(int(x) for x in totals)
        else:
            # Grid prints "Total: 720 | Valid: ..." per phase.
            grid_matches = re.findall(r"Total:\s*(\d+)\s*\|", text)
            if grid_matches:
                sims = sum(int(x) for x in grid_matches)
            else:
                # Last resort: use budget cap, doubled for disaggregated runs
                m_s = re.search(r"Total budget:\s*(\d+)", text)
                if m_s:
                    sims = int(m_s.group(1)) * (2 if "DISAGGREGATED" in text or "PREFILL optimization" in text else 1)

    return {"ttft": ttft, "tpot": tpot, "sims": sims if sims and sims > 0 else None}


def collect_all():
    """Parse all logs into structured results dict."""
    # results[config_id][model_key][method] = {ttft, tpot, sims}
    results = {}

    for logfile in sorted(LOGDIR.glob("*.log")):
        name = logfile.stem
        data = parse_log(logfile)
        if data["ttft"] is None and data["tpot"] is None:
            continue

        # Determine method
        if "_agent_" in name:
            method = "NPUAgent"
        elif "_baseline_bo_" in name:
            method = "BO"
        elif "_baseline_es_" in name:
            method = "CMA-ES"
        elif "_baseline_greedy_" in name:
            method = "Greedy"
        elif "_baseline_grid_" in name:
            method = "Grid"
        elif "_chatdse_" in name:
            method = "ChatDSE"
        elif "_llm_dse_" in name:
            method = "LLM-DSE"
        elif "_llm_box_" in name:
            method = "LLM-Box"
        else:
            continue

        # Parse config from filename: ..._tp4_pp2_bs16_300W.log
        m_cfg = re.search(r"tp(\d+)_pp(\d+)_bs(\d+)_(\d+)W", name)
        if not m_cfg:
            # Try C-style: _C5_
            m_c = re.search(r"_C(\d+)_", name)
            if m_c:
                config_id = f"C{m_c.group(1)}"
            else:
                continue
        else:
            tp, pp, bs, pw = m_cfg.groups()
            # Map to config ID by matching
            config_id = _find_config_id(name, int(tp), int(pp), int(bs), int(pw))

        if config_id is None:
            continue

        # Model
        model_key = None
        for mk, _ in MODELS_DISPLAY:
            if name.startswith(mk):
                model_key = mk
                break
        if not model_key:
            continue

        results.setdefault(config_id, {}).setdefault(model_key, {})[method] = data

    return results


def _find_config_id(name, tp, pp, bs, pw):
    """Map tp/pp/bs/pw to a config ID."""
    # Determine group from model name
    for mk, _ in MODELS_DISPLAY:
        if name.startswith(mk):
            if mk in ("llama3-8b", "llama2-13b"):
                group = "small"
            elif mk == "deepseekv2-236b":
                group = "dsv2"
            else:
                group = "large"
            break
    else:
        return None

    for cid, descs in CONFIGS_DESC.items():
        desc = descs[group]
        m = re.match(r"TP=(\d+), PP=(\d+), BS=(\d+), (\d+)W", desc)
        if m and int(m.group(1)) == tp and int(m.group(2)) == pp and int(m.group(3)) == bs and int(m.group(4)) == pw:
            return cid

    return None


def fmt_ttft(val):
    if val is None:
        return "---"
    if val < 1:
        return f"{val:.4f}"
    return f"{val:.2f}"


def fmt_tpot(val):
    if val is None:
        return "---"
    return f"{val:.2f}"


def fmt_sims(val, method=None):
    # ChatDSE relies entirely on LLM prediction; the single ground-truth
    # simulation per phase is verification, not search. Reporting it as a
    # sim count is misleading next to other methods' 100+ search calls.
    if method == "ChatDSE":
        return r"\textbackslash{}"
    if val is None or val == 0:
        return "---"
    return str(val)


def generate_subtable(config_id, config_results):
    """Generate one LaTeX table for a specific config."""
    # Build caption
    desc_parts = []
    for group in ["small", "large", "dsv2"]:
        desc = CONFIGS_DESC[config_id][group]
        if group == "small":
            desc_parts.append(f"Small models (8B, 13B): 8 chips, {desc}")
        elif group == "large":
            desc_parts.append(f"Large models (70B, 405B, 671B): 64 chips, {desc}")
        else:
            desc_parts.append(f"DSv2-236B: 16 chips, {desc}")

    # Subtable captions only show the per-deployment varying info
    # (power budget). The 4 fixed constraints (HBM cap/bw, ICI bw, DCN bw)
    # are stated once in the main text §4 (experiment setup).
    caption = f"Config {config_id}. {'; '.join(desc_parts)}."

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"    \centering")
    lines.append(f"    \\caption{{{caption}}}")
    lines.append(f"    \\label{{tab:{config_id.lower()}}}")
    lines.append(r"    \small")
    lines.append(r"    \setlength{\tabcolsep}{4pt}")
    lines.append(r"    \resizebox{\linewidth}{!}{")
    lines.append(r"    \begin{tabular}{c lcc|cc|cc|cc|cc|cc}")
    lines.append(r"        \toprule")
    lines.append(r"        \multirow{2}{*}{} & \multirow{2}{*}{Method}")

    for _, display in MODELS_DISPLAY:
        lines.append(f"        & \\multicolumn{{2}}{{c|}}{{{display}}}")
    # Fix last column (no |)
    last_line = lines[-1]
    lines[-1] = last_line.replace("c|}", "c}")
    lines.append(r" \\")

    lines.append(r"        &")
    for _ in MODELS_DISPLAY:
        lines.append(r"        & TTFT & Sims")
    lines.append(r" \\")
    lines.append(r"        \midrule")
    lines.append("")

    # Prefill section
    lines.append(r"        \multirow{8}{*}{\rotatebox{90}{\textbf{Prefill}}}")

    # Find best TTFT per model for bolding
    best_ttft = {}
    for mk, _ in MODELS_DISPLAY:
        vals = []
        for method in METHODS:
            d = config_results.get(mk, {}).get(method, {})
            if d.get("ttft") is not None:
                vals.append(d["ttft"])
        best_ttft[mk] = min(vals) if vals else None

    for method in METHODS:
        line_parts = [f"        & {method}"]
        for mk, _ in MODELS_DISPLAY:
            d = config_results.get(mk, {}).get(method, {})
            ttft = d.get("ttft")
            sims = d.get("sims")
            ttft_str = fmt_ttft(ttft)
            if ttft is not None and best_ttft.get(mk) is not None and abs(ttft - best_ttft[mk]) < 0.0001:
                ttft_str = f"\\textbf{{{ttft_str}}}"
            line_parts.append(f"        & {ttft_str} & {fmt_sims(sims, method)}")
        lines.append("\n".join(line_parts) + r" \\")
        lines.append("")

    # Divider for decode
    lines.append(r"        \midrule")
    lines.append(r"        & Method")
    for _ in MODELS_DISPLAY:
        lines.append(r"        & TPOT & Sims")
    lines.append(r" \\")
    lines.append(r"        \midrule")
    lines.append("")

    # Decode section
    lines.append(r"        \multirow{8}{*}{\rotatebox{90}{\textbf{Decode}}}")

    best_tpot = {}
    for mk, _ in MODELS_DISPLAY:
        vals = []
        for method in METHODS:
            d = config_results.get(mk, {}).get(method, {})
            if d.get("tpot") is not None:
                vals.append(d["tpot"])
        best_tpot[mk] = min(vals) if vals else None

    for method in METHODS:
        line_parts = [f"        & {method}"]
        for mk, _ in MODELS_DISPLAY:
            d = config_results.get(mk, {}).get(method, {})
            tpot = d.get("tpot")
            sims = d.get("sims")
            tpot_str = fmt_tpot(tpot)
            if tpot is not None and best_tpot.get(mk) is not None and abs(tpot - best_tpot[mk]) < 0.01:
                tpot_str = f"\\textbf{{{tpot_str}}}"
            line_parts.append(f"        & {tpot_str} & {fmt_sims(sims, method)}")
        lines.append("\n".join(line_parts) + r" \\")
        lines.append("")

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}}")
    lines.append(r"\end{table}")
    lines.append("")

    return "\n".join(lines)


def generate_main_table(all_results):
    """Generate win rate summary table with both prefill and decode sections."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"    \centering")
    lines.append(r"    \caption{Win rate and average simulator calls across 16 deployment configurations per model. Small models (8B, 13B) use 8 chips; large models (70B--671B) use 64 chips (DSv2-236B uses 16 chips). Win rate counts configurations where the method achieves the best (or tied-for-best) latency.}")
    lines.append(r"    \label{tab:main-winrate}")
    lines.append(r"    \small")
    lines.append(r"    \setlength{\tabcolsep}{4pt}")
    lines.append(r"    \resizebox{\linewidth}{!}{")
    lines.append(r"    \begin{tabular}{c lcc|cc|cc|cc|cc|cc}")
    lines.append(r"        \toprule")
    lines.append(r"        \multirow{2}{*}{} & \multirow{2}{*}{Method}")

    for _, display in MODELS_DISPLAY:
        lines.append(f"        & \\multicolumn{{2}}{{c|}}{{{display}}}")
    last_line = lines[-1]
    lines[-1] = last_line.replace("c|}", "c}")
    lines.append(r" \\")

    lines.append(r"        &")
    for _ in MODELS_DISPLAY:
        lines.append(r"        & Win & Sims")
    lines.append(r" \\")
    lines.append(r"        \midrule")
    lines.append("")

    # Helper: compute per-(method, model, metric) win count and avg sims
    def compute_cell(method, mk, metric):
        wins = total = total_sims = sim_count = 0
        for cid in [f"C{i}" for i in range(1, 17)]:
            cfg_data = all_results.get(cid, {}).get(mk, {})
            method_data = cfg_data.get(method, {})
            if method_data.get(metric) is None:
                continue
            total += 1
            best_val = min(
                (cfg_data.get(m, {}).get(metric, float("inf")) for m in METHODS),
                default=float("inf"))
            eps = 0.001 if metric == "ttft" else 0.01
            if abs(method_data[metric] - best_val) < eps:
                wins += 1
            if method_data.get("sims"):
                total_sims += method_data["sims"]
                sim_count += 1
        avg_sims = (total_sims / sim_count) if sim_count > 0 else None
        return wins, total, avg_sims

    def styled_win(wins, total, rank):
        """rank: 1=best (bold), 2=second (underline), else plain. wins/total fixed denom 16."""
        s = f"{wins}/{total}" if total > 0 else "---"
        if total == 0:
            return s
        if rank == 1:
            return f"\\textbf{{{s}}}"
        if rank == 2:
            return f"\\underline{{{s}}}"
        return s

    def styled_sims(avg_sims, method, rank_low):
        """rank_low: 1=lowest sims (bold), 2=second-lowest (underline)."""
        if method == "ChatDSE":
            return r"\textbackslash{}"
        if avg_sims is None:
            return "---"
        s = f"{avg_sims:.0f}"
        if rank_low == 1:
            return f"\\textbf{{{s}}}"
        if rank_low == 2:
            return f"\\underline{{{s}}}"
        return s

    def column_ranks(values, exclude_methods=()):
        """Given dict {method: value}, return {method: rank}.
        Methods in exclude_methods are not ranked (e.g., ChatDSE for sims)."""
        items = [(m, v) for m, v in values.items()
                 if v is not None and m not in exclude_methods]
        items.sort(key=lambda x: x[1])
        ranks = {}
        prev = None; rank = 0; same_rank = 0
        for i, (m, v) in enumerate(items):
            if prev is None or abs(v - prev) > 1e-9:
                rank = i + 1
            ranks[m] = rank
            prev = v
        return ranks

    # Pre-compute all cells
    cell_data = {}  # cell_data[(method, mk, metric)] = (wins, total, avg_sims)
    for method in METHODS:
        for mk, _ in MODELS_DISPLAY:
            for metric in ("ttft", "tpot"):
                cell_data[(method, mk, metric)] = compute_cell(method, mk, metric)

    # Per-(model, metric) ranks: highest wins → rank 1, highest 2nd → rank 2
    # For sims, lowest avg → rank 1 (good), excluding ChatDSE
    win_ranks = {}   # win_ranks[(mk, metric)] = {method: rank}
    sims_ranks = {}
    for mk, _ in MODELS_DISPLAY:
        for metric in ("ttft", "tpot"):
            wins_by_method = {m: -cell_data[(m, mk, metric)][0]
                              for m in METHODS
                              if cell_data[(m, mk, metric)][1] > 0}
            win_ranks[(mk, metric)] = column_ranks(wins_by_method)
            sims_by_method = {m: cell_data[(m, mk, metric)][2]
                              for m in METHODS
                              if cell_data[(m, mk, metric)][2] is not None}
            sims_ranks[(mk, metric)] = column_ranks(
                sims_by_method, exclude_methods=("ChatDSE",))

    # Prefill section
    lines.append(r"        \multirow{8}{*}{\rotatebox{90}{\textbf{Prefill}}}")
    for method in METHODS:
        line_parts = [f"        & {method}"]
        for mk, _ in MODELS_DISPLAY:
            wins, total, avg_sims = cell_data[(method, mk, "ttft")]
            wr = win_ranks[(mk, "ttft")].get(method, None)
            sr = sims_ranks[(mk, "ttft")].get(method, None)
            line_parts.append(
                f"        & {styled_win(wins, total, wr)} "
                f"& {styled_sims(avg_sims, method, sr)}"
            )
        lines.append("\n".join(line_parts) + r" \\")
        lines.append("")

    # Divider for decode
    lines.append(r"        \midrule")
    lines.append(r"        & Method")
    for _ in MODELS_DISPLAY:
        lines.append(r"        & Win & Sims")
    lines.append(r" \\")
    lines.append(r"        \midrule")
    lines.append("")

    # Decode section
    lines.append(r"        \multirow{8}{*}{\rotatebox{90}{\textbf{Decode}}}")

    for method in METHODS:
        line_parts = [f"        & {method}"]
        for mk, _ in MODELS_DISPLAY:
            wins, total, avg_sims = cell_data[(method, mk, "tpot")]
            wr = win_ranks[(mk, "tpot")].get(method, None)
            sr = sims_ranks[(mk, "tpot")].get(method, None)
            line_parts.append(
                f"        & {styled_win(wins, total, wr)} "
                f"& {styled_sims(avg_sims, method, sr)}"
            )
        lines.append("\n".join(line_parts) + r" \\")
        lines.append("")

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    print("Collecting results...")
    results = collect_all()
    print(f"Found data for {len(results)} configs")

    for cid in sorted(results.keys()):
        models_with_data = list(results[cid].keys())
        methods_per = {mk: list(results[cid][mk].keys()) for mk in models_with_data}
        print(f"  {cid}: {len(models_with_data)} models, methods: {methods_per}")

    # Save raw results
    with open("experiment_480/all_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Generate tables
    output_lines = []

    # Main table (win rate)
    output_lines.append("% === MAIN TABLE: Win Rate Summary ===")
    output_lines.append(generate_main_table(results))
    output_lines.append("")
    output_lines.append(r"\clearpage")
    output_lines.append("")

    # 16 sub-tables
    for i in range(1, 17):
        cid = f"C{i}"
        config_results = results.get(cid, {})
        output_lines.append(f"% === CONFIG {cid} ===")
        output_lines.append(generate_subtable(cid, config_results))
        output_lines.append("")

    output_path = "/mnt/data1/yh84/chip/current_result.tex"
    with open(output_path, "w") as f:
        f.write("\n".join(output_lines))

    print(f"\nGenerated 17 tables → {output_path}")


if __name__ == "__main__":
    main()
