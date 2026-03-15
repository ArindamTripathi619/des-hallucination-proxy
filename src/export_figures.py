"""Export figures programmatically from CSV tables into paper/figures/.

Generates:
  - paper/figures/calibration_curve.png  (and .pdf)
  - paper/figures/auroc_vs_n_models.png  (and .pdf)
  - paper/figures/alpha_sensitivity.png  (and .pdf)
  - paper/figures/lomo_auroc.png          (and .pdf)
  - paper/figures/embedding_ablation.png (and .pdf)

Usage:
  python src/export_figures.py --expanded
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from utils import OUTPUTS_TABLES


OUT_DIR = pathlib.Path(__file__).parent.parent / "paper" / "figures"
PNG_DIR = OUT_DIR / "png"
PDF_DIR = OUT_DIR / "pdf"
PNG_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name: str):
    png = PNG_DIR / f"{name}.png"
    pdf = PDF_DIR / f"{name}.pdf"
    fig.savefig(png, bbox_inches="tight", dpi=200)
    fig.savefig(pdf, bbox_inches="tight", dpi=200)


def plot_auroc_vs_n():
    f = OUTPUTS_TABLES / "auroc_vs_n_models.csv"
    if not f.exists():
        print(f"[SKIP] {f} not found (auroc vs n).")
        return
    df = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.errorbar(df["N_models"], df["Mean_AUROC"], yerr=df["Std_AUROC"], 
                marker="o", color="forestgreen", capsize=5, linewidth=2)
    
    # Add labels with bounding boxes
    for x, y in zip(df["N_models"], df["Mean_AUROC"]):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8, ec='forestgreen'))
    
    ax.set_xlabel("Number of models (N)")
    ax.set_ylabel("Mean AUROC")
    ax.set_title("Detection Performance vs. Ensemble Size", fontweight='bold', pad=25)
    
    # Dynamic limits to prevent cutoff
    ymin = df["Mean_AUROC"].min()
    ymax = df["Mean_AUROC"].max()
    ax.set_ylim(ymin - 0.05, ymax + 0.05)
    ax.grid(linestyle='--', alpha=0.6)
    
    save_fig(fig, "auroc_vs_n_models")
    plt.close(fig)


def plot_alpha_sensitivity():
    f = OUTPUTS_TABLES / "alpha_sensitivity.csv"
    if not f.exists():
        print(f"[SKIP] {f} not found (alpha sensitivity).")
        return
    df = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(df["Alpha"], df["AUROC"], marker="o", color="darkorchid", linewidth=2)
    
    # Add labels
    for x, y in zip(df["Alpha"], df["AUROC"]):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8, ec='darkorchid'))
    
    ax.set_xlabel("Alpha (surface weight)")
    ax.set_ylabel("AUROC")
    ax.set_title("Impact of Surface vs. Hidden Weight (Alpha)", fontweight='bold', pad=25)
    
    # Dynamic limits to encompass full range
    ymin = df["AUROC"].min()
    ymax = df["AUROC"].max()
    ax.set_ylim(ymin - 0.05, ymax + 0.05)
    ax.grid(linestyle='--', alpha=0.6)
    
    save_fig(fig, "alpha_sensitivity")
    plt.close(fig)


def plot_lomo():
    f = OUTPUTS_TABLES / "robustness_lomo.csv"
    if not f.exists():
        print(f"[SKIP] {f} not found (LOMO).")
        return
    df = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(8, 5))
    df_sorted = df.sort_values("AUROC")
    
    bars = ax.barh(df_sorted["Dropped_Model"], df_sorted["AUROC"], color="skyblue", edgecolor="navy")
    
    # Add labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', va='center', fontsize=9, fontweight='bold')
    
    ax.set_xlabel("AUROC after dropping model")
    ax.set_title("LOMO Stability: AUROC per Dropped Model (9-Model Expansion)", fontweight='bold')
    
    # Zoom in to see the differences
    ax.set_xlim(0.94, 0.96)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    save_fig(fig, "lomo_auroc")
    plt.close(fig)


def plot_embedding_ablation():
    f = OUTPUTS_TABLES / "robustness_embedding_ablation.csv"
    if not f.exists():
        print(f"[SKIP] {f} not found (embedding ablation).")
        return
    df = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(8, 5))
    
    y_err = (df["CI_upper"] - df["CI_lower"]) / 2
    bars = ax.bar(df["Embedding_Model"], df["AUROC"], yerr=y_err, 
                  capsize=8, color="lightcoral", edgecolor="darkred", alpha=0.8)
    
    # Add labels atop bars
    for bar, val in zip(bars, df["AUROC"]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel("AUROC")
    ax.set_title("Embedding Model Ablation (Robustness Check)", fontweight='bold')
    
    # Zoom in to see the differences
    ax.set_ylim(0.90, 0.96)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.xticks(rotation=15)
    save_fig(fig, "embedding_ablation")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Export figures programmatically")
    parser.add_argument("--expanded", action="store_true", help="Use expanded results (not used here)")
    args = parser.parse_args()

    print("Exporting figures to:", OUT_DIR)
    plot_auroc_vs_n()
    plot_alpha_sensitivity()
    plot_lomo()
    plot_embedding_ablation()
    print("Done.")


if __name__ == "__main__":
    main()
