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


def leakage_figure():
    g, n = SUMMARY["b0_mtl"], SUMMARY["b0_mtl_naive"]
    labels = ["HSV+LR\nfreshness", "HSV+LR\ntype", "B0 MTL\nfreshness", "B0 MTL\ntype"]
    naive = [BASELINE["split_naive"]["freshness"]["accuracy"], BASELINE["split_naive"]["produce_type"]["accuracy"],
             n["freshness_accuracy"]["mean"], n["produce_type_accuracy"]["mean"]]
    grouped = [BASELINE["split"]["freshness"]["accuracy"], BASELINE["split"]["produce_type"]["accuracy"],
               g["freshness_accuracy"]["mean"], g["produce_type_accuracy"]["mean"]]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    w = 0.38
    ax.bar([i - w / 2 for i in x], [100 * v for v in naive], w, label="Per-file split", color="#c8553d")
    ax.bar([i + w / 2 for i in x], [100 * v for v in grouped], w, label="Grouped split", color="#2f6690")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("Test accuracy (%)", fontsize=7)
    ax.set_ylim(min(100 * v for v in grouped) - 10, 100)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "leakage.pdf")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    sess_table, sess_macros = session_table()
    (OUT / "numbers.tex").write_text(
        "% Generated by paper/make_tables.py - do not edit\n"
        + dataset_numbers() + results_numbers() + sess_macros
    )
    (OUT / "tables.tex").write_text(
        "% Generated by paper/make_tables.py - do not edit\n"
        + split_table() + leakage_table() + sess_table + main_table() + ood_table()
    )
    leakage_figure()
    print(f"wrote {OUT / 'numbers.tex'}, {OUT / 'tables.tex'}, {FIG / 'leakage.pdf'}")
