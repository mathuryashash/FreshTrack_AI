"""Generate LaTeX macros, tables and the leakage figure for the paper.

Every number in paper/freshtrack_ieee.tex comes from here, which reads only
results/summary.json, results/baseline.json, models/runs/*/metrics.json and the
metadata files. Run after src.training.run_experiment:

    python paper/make_tables.py   # -> paper/generated/{numbers,tables}.tex, paper/figures/leakage.pdf
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data.build_splits import session_key, source_key as source_key_fn  # noqa: E402

SESSION_META = json.loads((ROOT / "data/metadata_session.json").read_text())
OUT = ROOT / "paper" / "generated"
FIG = ROOT / "paper" / "figures"
SUMMARY = json.loads((ROOT / "results/summary.json").read_text())
BASELINE = json.loads((ROOT / "results/baseline.json").read_text())
META = json.loads((ROOT / "data/metadata_v2.json").read_text())
EXT = json.loads((ROOT / "data/metadata_external.json").read_text())

PRETTY = {
    "b0_mtl": "EfficientNet-B0, multi-task",
    "b0_fresh": "EfficientNet-B0, freshness only",
    "b0_type": "EfficientNet-B0, type only",
    "mnv3_mtl": "MobileNetV3-L, multi-task",
    "b0_mtl_naive": "EfficientNet-B0, multi-task (per-file split)",
    "b0_mtl_session": "EfficientNet-B0, multi-task (held-out sessions)",
}
SCORES = {
    "neg_entropy_freshness": "Neg. entropy, freshness (v1 gate)",
    "msp_freshness": "MSP, freshness",
    "msp_produce_type": "MSP, produce type",
    "energy_produce_type": "Energy, produce type",
}


def pct(m, key, std=True):
    if key not in m:
        return "--"
    v = m[key]
    return f"{100 * v['mean']:.1f} $\\pm$ {100 * v['std']:.1f}" if std else f"{100 * v['mean']:.1f}"


def macro(name, value):
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def dataset_numbers():
    imgs = META["images"]
    groups = defaultdict(set)
    for r in imgs:
        groups[r["group_id"]].add(r["split"])
    train_groups_naive = {r["group_id"] for r in imgs if r["split_naive"] == "train"}
    naive_test = [r for r in imgs if r["split_naive"] == "test"]
    leak = sum(r["group_id"] in train_groups_naive for r in naive_test) / len(naive_test)
    split_counts = Counter(r["split"] for r in imgs)
    test_groups = len({r["group_id"] for r in imgs if r["split"] == "test"})
    roles = Counter(r["role"] for r in EXT["images"])
    out = ""
    out += macro("NImages", f"{len(imgs):,}")
    out += macro("NGroups", f"{len(groups):,}")
    out += macro("ImgsPerGroup", f"{len(imgs) / len(groups):.1f}")
    out += macro("NTrain", f"{split_counts['train']:,}")
    out += macro("NVal", f"{split_counts['val']:,}")
    out += macro("NTest", f"{split_counts['test']:,}")
    out += macro("NTestGroups", f"{test_groups:,}")
    out += macro("NaiveLeakPct", f"{100 * leak:.1f}")
    out += macro("NCross", f"{roles['cross_dataset']:,}")
    out += macro("NNearOOD", f"{roles['near_ood']:,}")
    out += macro("NExtDropped", f"{EXT['dropped_near_duplicates_of_main']}")
    out += macro("SplitSeed", f"{META['split_seed']}")

    def skey(r):
        return (r["produce_type"], r["freshness"], session_key(Path(r["image_path"]).name))

    sessions = {skey(r) for r in imgs}
    train_sessions = {skey(r) for r in imgs if r["split"] == "train"}
    test = [r for r in imgs if r["split"] == "test"]
    shared = sum(skey(r) in train_sessions for r in test) / len(test)
    out += macro("NSessions", f"{len(sessions)}")
    out += macro("SessionSharePct", f"{100 * shared:.1f}")
    single = sum(
        1 for k in {(r["produce_type"], r["freshness"]) for r in imgs}
        if len({s for s in sessions if s[:2] == k}) == 1
    )
    out += macro("NSingleSessionStrata", f"{single}")
    held = SESSION_META["held_out_sessions"]
    out += macro("NHeldOutStrata", f"{len(held)}")
    sc = Counter(r["split_session"] for r in SESSION_META["images"])
    out += macro("NSessTrain", f"{sc['train']:,}")
    out += macro("NSessVal", f"{sc['val']:,}")
    out += macro("NSessTest", f"{sc['test']:,}")
    out += macro("NSessExcluded", f"{sc['excluded']:,}")
    fname_groups = {(r["produce_type"], source_key_fn(Path(r["image_path"]).name)) for r in imgs}
    out += macro("NFilenameGroups", f"{len(fname_groups):,}")
    labels = defaultdict(set)
    sizes = Counter(r["group_id"] for r in imgs)
    for r in imgs:
        labels[r["group_id"]].add(r["freshness"])
    mixed = [g for g, l in labels.items() if len(l) > 1]
    out += macro("NMixedClusters", f"{len(mixed)}")
    out += macro("NMixedClusterImgs", f"{sum(sizes[g] for g in mixed):,}")
    out += macro("MaxClusterSize", f"{max(sizes.values()):,}")
    return out


def _pooled_strata(exp, task, strata):
    """Accuracy over the given strata, pooled over all seeds of an experiment."""
    correct = total = 0
    for p in sorted((ROOT / "models/runs").glob(f"{exp}_s[0-9]*/metrics.json")):
        per = json.loads(p.read_text())["tasks"][task]["per_stratum"]
        for s in strata:
            if s in per:
                correct += per[s][0]
                total += per[s][1]
    return correct / total if total else float("nan"), total


def pretty_session(key):
    """'img_20200901' -> 'IMG 2020-09-01'; 'screen shot 2018-06-08' -> 'Screenshots 2018-06-08'."""
    import re
    m = re.match(r"img_(\d{4})(\d\d)(\d\d)", key)
    if m:
        return f"IMG {m[1]}-{m[2]}-{m[3]}"
    if key.startswith("screen shot "):
        return "Screenshots " + key[len("screen shot "):]
    if key.startswith("whatsapp image "):
        return "WhatsApp " + key[len("whatsapp image "):]
    return {"day": "Day series", "dsc": "DSC series"}.get(key, key.capitalize() + " series")


def session_table():
    """Same strata, two protocols: photo-grouped test vs held-out-session test.

    Strata absent from the photo-grouped test (large groups can empty a
    stratum) are shown as '--' and excluded from the pooled comparison.
    """
    held = SESSION_META["held_out_sessions"]
    strata = sorted(held)
    rows, macros, common = [], "", []
    for s in strata:
        t, f = s.split("|")
        g_acc, g_n = _pooled_strata("b0_mtl", "freshness", [s])
        s_acc, s_n = _pooled_strata("b0_mtl_session", "freshness", [s])
        if g_n:
            common.append(s)
        g_cell = f"{100 * g_acc:.1f}" if g_n else "--"
        rows.append(
            f"{t.replace('_', ' ').capitalize()} ({f.lower()}) & {pretty_session(held[s])} & {s_n // 3} "
            f"& {g_cell} & {100 * s_acc:.1f} \\\\"
        )
        name = "".join(w.capitalize() for w in t.split("_")) + f
        macros += macro(f"SessHeld{name}", f"{100 * s_acc:.1f}")
    # Unweighted mean over strata, so both protocols weight strata equally.
    for task, ttag in [("freshness", "Fresh"), ("produce_type", "Type")]:
        g = [_pooled_strata("b0_mtl", task, [c])[0] for c in common]
        h = [_pooled_strata("b0_mtl_session", task, [c])[0] for c in common]
        macros += macro(f"SessStrataGrouped{ttag}Acc", f"{100 * sum(g) / len(g):.1f}")
        macros += macro(f"SessStrataHeld{ttag}Acc", f"{100 * sum(h) / len(h):.1f}")
    macros += macro("NSessCommonStrata", str(len(common)))
    table = (
        "\\begin{table}[t]\n\\centering\n\\caption{Freshness accuracy (\\%, pooled over three seeds) per "
        "produce/freshness stratum: photo-grouped test (sessions seen in training) versus held-out-session "
        "test (an unseen capture session). $n$: held-out test images; --: stratum absent from the "
        "photo-grouped test.}\n\\label{tab:session}\n\\footnotesize\n\\setlength{\\tabcolsep}{2.5pt}\n"
        "\\begin{tabular}{llrcc}\n\\toprule\n"
        "Stratum & Held-out session & $n$ & Grouped & Held-out \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    return table, macros


def results_numbers():
    out = ""
    for key, tag in [("b0_mtl", "Mtl"), ("b0_fresh", "Fresh"), ("b0_type", "Type"),
                     ("mnv3_mtl", "Mnv"), ("b0_mtl_naive", "Naive"), ("b0_mtl_session", "Sess")]:
        m = SUMMARY.get(key, {})
        for metric, mtag in [("freshness_accuracy", "FreshAcc"), ("freshness_macro_f1", "FreshFone"),
                             ("produce_type_accuracy", "TypeAcc"), ("produce_type_macro_f1", "TypeFone"),
                             ("cross_produce_type_accuracy", "CrossTypeAcc"),
                             ("cross_share_predicted_fresh", "CrossShareFresh"),
                             ("freshness_ece", "FreshEce")]:
            if metric in m:
                out += macro(f"{tag}{mtag}", pct(m, metric, std=False))
                out += macro(f"{tag}{mtag}Std", f"{100 * m[metric]['std']:.1f}")
        if "cpu_latency_ms_batch1" in m:
            out += macro(f"{tag}CpuMs", f"{m['cpu_latency_ms_batch1']['mean']:.1f}")
            out += macro(f"{tag}Params", f"{m['params_millions']:.2f}")
        for gate in ("test_id_accept_rate", "near_ood_reject_rate", "far_ood_reject_rate"):
            k = f"gate_{gate}"
            if k in m:
                name = "".join(w.capitalize() for w in gate.split("_"))
                out += macro(f"{tag}Gate{name.replace('Ood', 'OOD')}", pct(m, k, std=False))
        for score, stag in [("energy_produce_type", "Energy"), ("neg_entropy_freshness", "Entropy"),
                            ("msp_freshness", "MspFresh")]:
            for ood, otag in [("near_ood", "Near"), ("far_ood", "Far")]:
                k = f"ood_{score}_{ood}_auroc"
                if k in m:
                    out += macro(f"{tag}{stag}{otag}Auroc", pct(m, k, std=False))
    for field, ftag in [("split", "Grouped"), ("split_naive", "Naive")]:
        for task, ttag in [("freshness", "Fresh"), ("produce_type", "Type")]:
            out += macro(f"Hist{ftag}{ttag}Acc", f"{100 * BASELINE[field][task]['accuracy']:.1f}")
    # 95% group-bootstrap CI of the seed-0 multi-task model
    s0 = json.loads((ROOT / "models/runs/b0_mtl_s0/metrics.json").read_text())
    for task, ttag in [("freshness", "Fresh"), ("produce_type", "Type")]:
        lo, hi = s0["tasks"][task]["accuracy_ci95"]
        out += macro(f"MtlSzero{ttag}Acc", f"{100 * s0['tasks'][task]['accuracy']:.1f}")
        out += macro(f"MtlSzero{ttag}CI", f"[{100 * lo:.1f}, {100 * hi:.1f}]")
    dep = json.loads((ROOT / "models/runs/mnv3_mtl_s1/metrics.json").read_text())["ood_gate"]
    out += macro("DeployGateAccept", f"{100 * dep['test_id_accept_rate']:.1f}")
    out += macro("DeployGateNear", f"{100 * dep['near_ood_reject_rate']:.1f}")
    out += macro("DeployGateFar", f"{100 * dep['far_ood_reject_rate']:.1f}")
    near = SUMMARY["mnv3_mtl"]["gate_near_ood_reject_rate"]["values"]
    out += macro("MnvGateNearMin", f"{100 * min(near):.1f}")
    out += macro("MnvGateNearMax", f"{100 * max(near):.1f}")
    for key, tag in [("b0_mtl", "Mtl"), ("mnv3_mtl", "Mnv")]:
        out += macro(f"{tag}CrossShareFreshExact", f"{100 * SUMMARY[key]['cross_share_predicted_fresh']['mean']:.1f}")
    epochs = [e for m in SUMMARY.values() for e in m["epochs_trained"]]
    out += macro("EpochsMin", str(min(epochs)))  # current_epoch after fit = epochs completed
    out += macro("EpochsMax", str(max(epochs)))
    return out


def split_table():
    types = META["produce_types"]
    c = Counter((r["produce_type"], r["freshness"], r["split"]) for r in META["images"])
    rows = []
    for t in types:
        cells = [f"{c[(t, f, s)]:,}" for s in ("train", "val", "test") for f in ("Fresh", "Stale")]
        rows.append(f"{t.replace('_', ' ').capitalize()} & " + " & ".join(cells) + r" \\")
    return (
        "\\begin{table}[t]\n\\centering\n\\caption{Grouped split: images per produce type, freshness and split.}\n"
        "\\label{tab:split}\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
        " & \\multicolumn{2}{c}{Train} & \\multicolumn{2}{c}{Val} & \\multicolumn{2}{c}{Test} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
        "Type & Fresh & Stale & Fresh & Stale & Fresh & Stale \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def main_table():
    rows = []
    for key in ("b0_mtl", "b0_fresh", "b0_type", "mnv3_mtl"):
        m = SUMMARY.get(key)
        if not m:
            continue
        rows.append(
            f"{PRETTY[key]} & {pct(m, 'freshness_accuracy')} & {pct(m, 'freshness_macro_f1')} & "
            f"{pct(m, 'produce_type_accuracy')} & {pct(m, 'produce_type_macro_f1')} & "
            f"{m['params_millions']:.2f} & {m['cpu_latency_ms_batch1']['mean']:.1f} \\\\"
        )
    b = BASELINE["split"]
    rows.append(
        f"HSV histogram + LR & {100 * b['freshness']['accuracy']:.1f} & {100 * b['freshness']['macro_f1']:.1f} & "
        f"{100 * b['produce_type']['accuracy']:.1f} & {100 * b['produce_type']['macro_f1']:.1f} & -- & -- \\\\"
    )
    return (
        "\\begin{table*}[t]\n\\centering\n"
        "\\caption{Test results on the photo-grouped split (capture sessions are still shared with training), mean $\\pm$ std over three seeds (\\%). "
        "CPU latency: batch 1, 4 threads, median of 200 interleaved runs.}\n\\label{tab:main}\n"
        "\\begin{tabular}{lcccccc}\n\\toprule\n"
        "Model & Fresh acc. & Fresh macro-F1 & Type acc. & Type macro-F1 & Params (M) & CPU (ms) \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    )


def leakage_table():
    g, n = SUMMARY.get("b0_mtl"), SUMMARY.get("b0_mtl_naive")
    rows = []
    for label, gf, nf, gt, nt in [
        ("HSV hist.+LR",
         BASELINE["split"]["freshness"]["accuracy"], BASELINE["split_naive"]["freshness"]["accuracy"],
         BASELINE["split"]["produce_type"]["accuracy"], BASELINE["split_naive"]["produce_type"]["accuracy"]),
        ("B0 multi-task",
         g["freshness_accuracy"]["mean"], n["freshness_accuracy"]["mean"],
         g["produce_type_accuracy"]["mean"], n["produce_type_accuracy"]["mean"]),
    ]:
        rows.append(
            f"{label} & {100 * nf:.1f} & {100 * gf:.1f} & {100 * (nf - gf):+.1f} & "
            f"{100 * nt:.1f} & {100 * gt:.1f} & {100 * (nt - gt):+.1f} \\\\"
        )
    return (
        "\\begin{table}[t]\n\\centering\n\\caption{Accuracy (\\%) under a per-file split versus the grouped split. "
        "$\\Delta$ = inflation caused by leakage.}\n\\label{tab:leak}\n\\footnotesize\n\\setlength{\\tabcolsep}{2.5pt}\n"
        "\\begin{tabular}{lcccccc}\n\\toprule\n"
        " & \\multicolumn{3}{c}{Freshness} & \\multicolumn{3}{c}{Produce type} \\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
        "Model & Per-file & Grouped & $\\Delta$ & Per-file & Grouped & $\\Delta$ \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def ood_table():
    rows = []
    for key in ("b0_mtl", "mnv3_mtl"):
        m = SUMMARY.get(key)
        if not m:
            continue
        rows.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{PRETTY[key]}}}}} \\\\")
        for score, label in SCORES.items():
            cells = []
            for ood in ("near_ood", "far_ood"):
                cells.append(pct(m, f"ood_{score}_{ood}_auroc"))
                cells.append(pct(m, f"ood_{score}_{ood}_fpr_at_95tpr"))
            rows.append(f"\\quad {label} & " + " & ".join(cells) + r" \\")
    return (
        "\\begin{table*}[t]\n\\centering\n\\caption{Out-of-distribution detection (\\%, mean $\\pm$ std over three seeds). "
        "Near-OOD: unseen produce classes from an external dataset; far-OOD: CIFAR-10 test images. "
        "Higher AUROC and lower FPR@95\\%TPR are better.}\n\\label{tab:ood}\n"
        "\\begin{tabular}{lcccc}\n\\toprule\n"
        " & \\multicolumn{2}{c}{Near-OOD (\\NNearOOD{} images)} & \\multicolumn{2}{c}{Far-OOD (2,000 images)} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
        "Score & AUROC & FPR@95 & AUROC & FPR@95 \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    )


# ── Figures: one IEEE column wide, Times-like font, legends outside the data ──
COLUMN_IN = 3.45
PALETTE = {"per_file": "#c0392b", "grouped": "#2471a3", "session": "#e1a100"}
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})


def leakage_figure():
    """Dot plot: accuracy of each model/task under the three protocols."""
    g, n, s = SUMMARY["b0_mtl"], SUMMARY["b0_mtl_naive"], SUMMARY["b0_mtl_session"]
    rows = [  # (label, per-file, grouped, held-out session or None)
        ("HSV + LR, type", BASELINE["split_naive"]["produce_type"]["accuracy"],
         BASELINE["split"]["produce_type"]["accuracy"], None),
        ("HSV + LR, freshness", BASELINE["split_naive"]["freshness"]["accuracy"],
         BASELINE["split"]["freshness"]["accuracy"], None),
        ("CNN, type", n["produce_type_accuracy"]["mean"], g["produce_type_accuracy"]["mean"],
         s["produce_type_accuracy"]["mean"]),
        ("CNN, freshness", n["freshness_accuracy"]["mean"], g["freshness_accuracy"]["mean"],
         s["freshness_accuracy"]["mean"]),
    ]
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 1.95), constrained_layout=True)
    for y, (_, pf, gr, se) in enumerate(rows):
        vals = [100 * v for v in (pf, gr, se) if v is not None]
        ax.plot([min(vals), max(vals)], [y, y], color="#b3b3b3", lw=1.2, zorder=1)
        # Hollow per-file ring stays visible when a grouped square sits on top of it
        ax.scatter(100 * pf, y, s=70, facecolors="none", edgecolors=PALETTE["per_file"], lw=1.4,
                   marker="o", zorder=4)
        ax.annotate(f"{100 * pf:.1f}", (100 * pf, y), xytext=(6, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=6.5, color=PALETTE["per_file"])
        ax.scatter(100 * gr, y, s=26, color=PALETTE["grouped"], marker="s", zorder=3)
        if se is not None:
            ax.scatter(100 * se, y, s=40, color=PALETTE["session"], marker="D", zorder=3)
            ax.annotate(f"{100 * se:.1f}", (100 * se, y), xytext=(0, -9), textcoords="offset points",
                        ha="center", va="top", fontsize=6.5)
        ax.annotate(f"{100 * gr:.1f}", (100 * gr, y), xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(76, 104.5)
    ax.set_xticks([80, 85, 90, 95, 100])
    ax.set_xlabel("Test accuracy (%)")
    ax.grid(axis="x", color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)
    handles = [
        plt.Line2D([], [], markeredgecolor=PALETTE["per_file"], markerfacecolor="none", marker="o",
                   ls="", mew=1.4, label="Per-file split"),
        plt.Line2D([], [], color=PALETTE["grouped"], marker="s", ls="", label="Photo-grouped"),
        plt.Line2D([], [], color=PALETTE["session"], marker="D", ls="", label="Unseen sessions"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False,
               handletextpad=0.3, columnspacing=1.0)
    fig.savefig(FIG / "leakage.pdf")
    plt.close(fig)


def session_figure():
    """Per-stratum freshness accuracy on unseen capture sessions (horizontal bars)."""
    held = SESSION_META["held_out_sessions"]
    rows = []
    for s in held:
        t, f = s.split("|")
        s_acc, s_n = _pooled_strata("b0_mtl_session", "freshness", [s])
        g_acc, g_n = _pooled_strata("b0_mtl", "freshness", [s])
        rows.append((f"{t.replace('_', ' ').capitalize()} ({f.lower()})", 100 * s_acc, s_n // 3,
                     100 * g_acc if g_n else None))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.35), constrained_layout=True)
    ys = range(len(rows))
    colors = ["#c0392b" if r[1] < 80 else "#7f8c8d" for r in rows]
    ax.barh(ys, [r[1] for r in rows], color=colors, height=0.62, zorder=2)
    for y, (_, acc, n_img, g) in zip(ys, rows):
        # Label at the left end of wide bars so it never meets the grouped-test tick
        # near the right end; narrow bars get the label just outside.
        inside = acc > 45
        ax.text(2 if inside else acc + 1.5, y, f"{acc:.1f}%  (n={n_img})",
                va="center", ha="left", fontsize=6.5,
                color="white" if inside else "black", zorder=3)
        if g is not None:
            ax.scatter(g, y, marker="|", s=90, color=PALETTE["grouped"], lw=1.6, zorder=4)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, 104)
    ax.set_xlabel("Freshness accuracy on the unseen session (%)")
    ax.grid(axis="x", color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#c0392b", label="Held-out session < 80%"),
        plt.Line2D([], [], color=PALETTE["grouped"], marker="|", ls="", markersize=9, mew=1.6,
                   label="Same stratum, photo-grouped test"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False,
               handletextpad=0.4, columnspacing=1.0)
    fig.savefig(FIG / "sessions.pdf")
    plt.close(fig)


def ood_figure():
    """Energy-score distributions of the deployed model, with the gate threshold."""
    import numpy as np

    d = json.loads((ROOT / "results/ood_scores_deployed.json").read_text())
    series = [
        ("in_distribution_test", "Supported produce (test)", PALETTE["grouped"]),
        ("near_ood", "Unseen produce (near-OOD)", PALETTE["session"]),
        ("far_ood", "CIFAR-10 (far-OOD)", PALETTE["per_file"]),
    ]
    lo = min(min(d[k]) for k, _, _ in series)
    hi = max(max(d[k]) for k, _, _ in series)
    bins = np.linspace(lo, hi, 45)
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.0), constrained_layout=True)
    for key, label, color in series:
        ax.hist(d[key], bins=bins, density=True, histtype="stepfilled", alpha=0.28, color=color)
        ax.hist(d[key], bins=bins, density=True, histtype="step", lw=1.1, color=color, label=label)
    ax.axvline(d["threshold"], color="black", lw=1.0, ls="--")
    ymax = ax.get_ylim()[1]
    ax.text(d["threshold"], ymax * 0.97, "  accept →", ha="left", va="top", fontsize=6.5)
    ax.text(d["threshold"], ymax * 0.97, "← reject  ", ha="right", va="top", fontsize=6.5)
    ax.set_xlabel("Energy score on the produce-type head")
    ax.set_ylabel("Density")
    ax.set_yticks([])
    fig.legend(loc="outside upper center", ncol=2, frameon=False, handlelength=1.4,
               columnspacing=1.0)
    fig.savefig(FIG / "ood.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    sess_table, sess_macros = session_table()
    (OUT / "numbers.tex").write_text(
        "% Generated by paper/make_tables.py - do not edit\n"
        + dataset_numbers() + results_numbers() + sess_macros
    )
    (OUT / "tables.tex").write_text(
        "% Generated by paper/make_tables.py - do not edit\n"
        + split_table() + leakage_table() + main_table() + ood_table()
    )
    (OUT / "session_table.tex").write_text(sess_table)  # kept for reference; the paper uses the figure
    mob = json.loads((ROOT / "results/mobile_metrics.json").read_text())
    (OUT / "app_numbers.tex").write_text(
        "% Generated by paper/make_tables.py from results/mobile_metrics.json - do not edit\n"
        + macro("AppApkArm", f"{mob['apk_mib']['arm64-v8a']:.1f}")
        + macro("AppApkArmSeven", f"{mob['apk_mib']['armeabi-v7a']:.1f}")
        + macro("AppOrtMb", f"{mob['apk_contents_mb_arm64']['onnxruntime_lib']:.1f}")
        + macro("AppModelMb", f"{mob['apk_contents_mb_arm64']['model']:.1f}")
        + macro("AppParityN", str(mob["parity"]["n_images"]))
        + macro("AppParityMaxDiff", f"{mob['parity']['max_abs_logit_diff']:.2f}")
        + macro("AppPipeMs", f"{mob['latency_ms']['pipeline_median_min']}--{mob['latency_ms']['pipeline_median_max']}")
        + macro("AppScanMs", str(mob["latency_ms"]["app_median"]))
        + macro("AppColdMs", f"{mob['cold_start_ms_median'] / 1000:.1f}")
        + macro("AppRamIdle", f"{mob['ram_pss_mb']['idle']:.0f}")
        + macro("AppRamLoaded", f"{mob['ram_pss_mb']['model_loaded']:.0f}")
        + macro("AppRamPeak", f"{mob['peak_rss_mb']:.0f}")
        + macro("AppScanKb", f"{mob['storage_per_scan_kb_mean']:.0f}")
    )
    FIG.mkdir(parents=True, exist_ok=True)
    leakage_figure()
    session_figure()
    ood_figure()
    print(f"wrote {OUT / 'numbers.tex'}, {OUT / 'tables.tex'}, figures leakage/sessions/ood.pdf")
