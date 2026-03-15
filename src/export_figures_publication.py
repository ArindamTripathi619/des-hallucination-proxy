"""Generate publication-style figures and update fig1..fig5 only when changed.

This script regenerates a subset of figures with a consistent publication style
and replaces existing `fig1_...`/`fig5_...` files only when the content differs.

Currently regenerates:
  - fig1_calibration_curves (from calibration_raw.csv)
  - fig5_auroc_vs_n_models (from auroc_vs_n_models.csv)

It will NOT touch fig2, fig3, fig4 because their generators are not implemented here.
"""
from __future__ import annotations

import hashlib
import shutil
import pathlib
import sys
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from utils import OUTPUTS_TABLES


FIG_DIR = pathlib.Path(__file__).parent.parent / "paper" / "figures"
PNG_DIR = FIG_DIR / "png"
PDF_DIR = FIG_DIR / "pdf"
PNG_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)


def _checksum(path: pathlib.Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _save_pub_fig(fig, base_name: str):
    png_new = PNG_DIR / f"{base_name}.new.png"
    pdf_new = PDF_DIR / f"{base_name}.new.pdf"
    fig.savefig(png_new, bbox_inches="tight", dpi=300)
    fig.savefig(pdf_new, bbox_inches="tight")
    plt.close(fig)
    return png_new, pdf_new


def make_calibration_pub():
    f = OUTPUTS_TABLES / "calibration_raw.csv"
    if not f.exists():
        print("[SKIP] calibration_raw.csv missing — cannot build fig1")
        return None
    df = pd.read_csv(f)
    sns.set(style="whitegrid", context="paper", rc={
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    palette = sns.color_palette("tab10")
    for _i, (src, grp) in enumerate(df.groupby("source")):
        ax.plot(grp["mean_DES"], grp["mean_error_rate"], marker="o", label=src, color=palette[_i % len(palette)])
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Mean DES")
    ax.set_ylabel("Mean error rate")
    ax.set_title("Calibration curves by dataset")
    ax.legend(frameon=False)
    return _save_pub_fig(fig, "fig1_calibration_curves")


def make_auroc_pub():
    f = OUTPUTS_TABLES / "auroc_vs_n_models.csv"
    if not f.exists():
        print("[SKIP] auroc_vs_n_models.csv missing — cannot build fig5")
        return None
    df = pd.read_csv(f)
    sns.set(style="whitegrid", context="paper", rc={
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.errorbar(df["N_models"], df["Mean_AUROC"], yerr=df["Std_AUROC"], marker="o", color=sns.color_palette("tab10")[2])
    ax.set_xlabel("Number of models")
    ax.set_ylabel("Mean AUROC")
    ax.set_title("AUROC vs. number of models")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(df["N_models"].tolist())
    return _save_pub_fig(fig, "fig5_auroc_vs_n_models")


def make_domain_heatmap():
    f = OUTPUTS_TABLES / "domain_sensitivity.csv"
    if not f.exists():
        print("[SKIP] domain_sensitivity.csv missing — cannot build fig2")
        return None
    df = pd.read_csv(f)
    # pivot to a single-row heatmap (domains as columns)
    vals = df.set_index("Domain")["AUROC"].sort_values()
    sns.set(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(8, 2.5))
    sns.heatmap(vals.to_frame().T, annot=True, fmt=".3f", cmap="viridis", cbar_kws={"orientation": "horizontal"}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Domain sensitivity (AUROC by domain)")
    return _save_pub_fig(fig, "fig2_domain_heatmap")


def make_roc_curves():
    # Load scored results and compute ROC curves for DES variants + SelfCheck
    scored = pathlib.Path(__file__).parent.parent / "data" / "results" / "scored_results.jsonl"
    if not scored.exists():
        print("[SKIP] scored_results.jsonl missing — cannot build fig3")
        return None
    import json
    from sklearn.metrics import roc_curve, auc
    from utils import extract_for_embedding, get_embedder

    records = []
    with open(scored) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    import numpy as np
    y_true = np.array([r.get("any_error") for r in records], dtype=float)
    des = np.array([r.get("DES") if r.get("DES") is not None else np.nan for r in records], dtype=float)
    surf = np.array([r.get("surface_DES") if r.get("surface_DES") is not None else np.nan for r in records], dtype=float)
    sem = np.array([r.get("semantic_DES") if r.get("semantic_DES") is not None else np.nan for r in records], dtype=float)

    # SelfCheck: surface and semantic using llama models
    llama_models = ["llama-large", "llama-small", "llama4-scout"]
    sc_surface = []
    sc_semantic = []
    embedder = get_embedder()
    for r in records:
        answers = [r.get("model_responses", {}).get(m, {}).get("response") for m in llama_models]
        answers = [a for a in answers if a]
        if len(answers) < 2:
            sc_surface.append(np.nan)
            sc_semantic.append(np.nan)
            continue
        # surface
        norms = [extract_for_embedding(a) for a in answers]
        normed = [n for n in norms if n is not None]
        if len(normed) < 2:
            sc_surface.append(np.nan)
        else:
            pairs = [(normed[i], normed[j]) for i in range(len(normed)) for j in range(i+1, len(normed))]
            disagree = sum(1 for a,b in pairs if a != b)
            sc_surface.append(disagree / len(pairs))
        # semantic
        texts = [extract_for_embedding(a) for a in answers]
        texts = [t for t in texts if t]
        if len(texts) < 2:
            sc_semantic.append(np.nan)
        else:
            embs = embedder.encode(texts, convert_to_numpy=True)
            emb_norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
            embs = embs / emb_norms
            sims = [np.dot(embs[i], embs[j]) for i in range(len(embs)) for j in range(i+1, len(embs))]
            sc_semantic.append(1.0 - float(np.mean(sims)))

    sc_surface = np.array(sc_surface, dtype=float)
    sc_semantic = np.array(sc_semantic, dtype=float)

    curves = [
        (des, "DES (combined)"),
        (surf, "DES (surface)"),
        (sem, "DES (semantic)"),
        (sc_surface, "SelfCheck (surface)"),
        (sc_semantic, "SelfCheck (semantic)"),
    ]

    sns.set(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    palette = sns.color_palette("tab10")

    # Load bootstrap CI bounds for DES combined (issue #20 fix)
    ci_lower_bound, ci_upper_bound = None, None
    ci_file = OUTPUTS_TABLES / "robustness_bootstrap_ci.csv"
    if ci_file.exists():
        import pandas as _pd
        ci_df = _pd.read_csv(ci_file)
        des_ci = ci_df[(ci_df["Method"] == "DES (combined)") & (ci_df["Dataset"] == "all")]
        if not des_ci.empty:
            ci_str = des_ci.iloc[0].get("AUROC_95CI", "")
            try:
                import re as _re
                nums = _re.findall(r"[\d.]+", str(ci_str))
                if len(nums) >= 2:
                    ci_lower_bound = float(nums[0])
                    ci_upper_bound = float(nums[1])
            except Exception:
                pass

    for i, (scores, label) in enumerate(curves):
        mask = ~np.isnan(y_true) & ~np.isnan(scores)
        if mask.sum() < 2 or len(np.unique(y_true[mask])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true[mask].astype(int), scores[mask])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})", color=palette[i % len(palette)])

        # Add 95% CI shading for DES combined using precomputed bootstrap bounds
        if label == "DES (combined)" and ci_lower_bound is not None and ci_upper_bound is not None:
            ci_mid = (ci_lower_bound + ci_upper_bound) / 2
            ci_half = (ci_upper_bound - ci_lower_bound) / 2
            tpr_lower = np.clip(tpr - ci_half, 0, 1)
            tpr_upper = np.clip(tpr + ci_half, 0, 1)
            ax.fill_between(fpr, tpr_lower, tpr_upper, alpha=0.15,
                            color=palette[i % len(palette)], label="95% CI (DES combined)")

    ax.plot([0,1], [0,1], color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves: DES variants and SelfCheck")
    ax.legend(frameon=False)
    return _save_pub_fig(fig, "fig3_roc_curves")


def make_architecture_gap_pub():
    f = OUTPUTS_TABLES / "table4_architecture_gap.csv"
    if not f.exists():
        print("[SKIP] table4_architecture_gap.csv missing — cannot build fig4")
        return None
    df = pd.read_csv(f)
    sns.set(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(8, 4))
    df_sorted = df.sort_values("AUROC")
    ax.barh(df_sorted["label"], df_sorted["AUROC"], color=sns.color_palette("tab10"))
    ax.set_xlabel("AUROC")
    ax.set_title("Architecture-gap: pairwise AUROC by pair")
    ax.set_xlim(0,1)
    return _save_pub_fig(fig, "fig4_architecture_gap")


def replace_if_changed(new_pdf_path: pathlib.Path, existing_base: str):
    # Compare new .pdf with existing existing_base.pdf; if different, replace
    existing_pdf = PDF_DIR / f"{existing_base}.pdf"
    existing_png = PNG_DIR / f"{existing_base}.png"
    
    # The new files were saved as base.new.pdf in PDF_DIR and base.new.png in PNG_DIR
    new_pdf = new_pdf_path
    new_png = PNG_DIR / f"{existing_base}.new.png"
    
    new_checksum = _checksum(new_pdf)
    old_checksum = _checksum(existing_pdf)
    if old_checksum is None:
        # No existing file — move new into place
        shutil.move(str(new_pdf), str(existing_pdf))
        shutil.move(str(new_png), str(existing_png))
        print(f"Created new figure: {existing_base}")
        return True
    if new_checksum != old_checksum:
        # Replace
        shutil.move(str(new_pdf), str(existing_pdf))
        shutil.move(str(new_png), str(existing_png))
        print(f"Updated figure: {existing_base}")
        return True
    else:
        # No change — remove new files
        new_pdf.unlink(missing_ok=True)
        new_png.unlink(missing_ok=True)
        print(f"No change for: {existing_base}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate publication-style figures and update fig1..fig5 if changed")
    parser.add_argument("--force", action="store_true", help="Force replacement of files regardless of checksum")
    args = parser.parse_args()

    updated = {}

    # fig1
    res1 = make_calibration_pub()
    if res1:
        if args.force:
            # move without checking
            shutil.move(str(res1[1]), str(PDF_DIR / "fig1_calibration_curves.pdf"))
            shutil.move(str(res1[0]), str(PNG_DIR / "fig1_calibration_curves.png"))
            updated["fig1"] = True
            print("Forced update: fig1_calibration_curves")
        else:
            updated["fig1"] = replace_if_changed(res1[1], "fig1_calibration_curves")

    # fig5
    res5 = make_auroc_pub()
    if res5:
        if args.force:
            shutil.move(str(res5[1]), str(PDF_DIR / "fig5_auroc_vs_n_models.pdf"))
            shutil.move(str(res5[0]), str(PNG_DIR / "fig5_auroc_vs_n_models.png"))
            updated["fig5"] = True
            print("Forced update: fig5_auroc_vs_n_models")
        else:
            updated["fig5"] = replace_if_changed(res5[1], "fig5_auroc_vs_n_models")

    # fig2
    res2 = make_domain_heatmap()
    if res2:
        if args.force:
            shutil.move(str(res2[1]), str(PDF_DIR / "fig2_domain_heatmap.pdf"))
            shutil.move(str(res2[0]), str(PNG_DIR / "fig2_domain_heatmap.png"))
            updated["fig2"] = True
            print("Forced update: fig2_domain_heatmap")
        else:
            updated["fig2"] = replace_if_changed(res2[1], "fig2_domain_heatmap")
    else:
        # report if not generated
        updated["fig2"] = None if not (PDF_DIR / "fig2_domain_heatmap.pdf").exists() else False

    # fig3
    res3 = make_roc_curves()
    if res3:
        if args.force:
            shutil.move(str(res3[1]), str(PDF_DIR / "fig3_roc_curves.pdf"))
            shutil.move(str(res3[0]), str(PNG_DIR / "fig3_roc_curves.png"))
            updated["fig3"] = True
            print("Forced update: fig3_roc_curves")
        else:
            updated["fig3"] = replace_if_changed(res3[1], "fig3_roc_curves")
    else:
        updated["fig3"] = None if not (PDF_DIR / "fig3_roc_curves.pdf").exists() else False

    # fig4
    res4 = make_architecture_gap_pub()
    if res4:
        if args.force:
            shutil.move(str(res4[1]), str(PDF_DIR / "fig4_architecture_gap.pdf"))
            shutil.move(str(res4[0]), str(PNG_DIR / "fig4_architecture_gap.png"))
            updated["fig4"] = True
            print("Forced update: fig4_architecture_gap")
        else:
            updated["fig4"] = replace_if_changed(res4[1], "fig4_architecture_gap")
    else:
        updated["fig4"] = None if not (PDF_DIR / "fig4_architecture_gap.pdf").exists() else False

    print("\nSummary (True=updated, False=no-change, None=not-generated):")
    for k, v in updated.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
