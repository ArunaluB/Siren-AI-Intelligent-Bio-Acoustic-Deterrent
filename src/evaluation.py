"""
Siren AI v2 — Enterprise Evaluation & Visualization Engine
═══════════════════════════════════════════════════════════════
ALL required graphs and metrics for research-grade evaluation.

Generates:
  1.  Learning Curve (train + val accuracy)
  2.  Loss / TD Error Curve
  3.  Reward Trend Graph
  4.  Policy Entropy Curve
  5.  Confusion Matrix (normalized + raw)
  6.  Action Distribution Histogram
  7.  State Visitation Frequency Heatmap
  8.  Class Imbalance Visualization
  9.  Overfitting Detection Graph
  10. Q-Value Stability Graph
  11. Multi-Seed Stability Plot

All saved as PNG at 300 DPI in /results folder.
Also exports: metrics.json, confusion_matrix.csv, entropy_log.csv, reward_log.csv
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Colab/server
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from typing import Dict, List, Tuple, Optional

from config import (
    MODES, MODE_INDEX, NUM_MODES, RESULTS_DIR,
    GRAPH_DPI, GRAPH_FORMAT, TARGET_ACCURACY,
    OVERFIT_TRAIN_THRESHOLD, OVERFIT_VAL_THRESHOLD,
    OVERFIT_GAP_THRESHOLD, UNDERFIT_THRESHOLD,
)


# ═══════════════════════════════════════════════════════════════
#  STYLE SETUP
# ═══════════════════════════════════════════════════════════════
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
COLORS = {
    "M0": "#2ecc71", "M1": "#3498db", "M2": "#e67e22", "M3": "#e74c3c",
    "train": "#2980b9", "val": "#e74c3c", "test": "#27ae60",
    "entropy": "#8e44ad", "reward": "#f39c12", "q_value": "#1abc9c",
}


def _save_fig(fig, name: str, results_dir: str = None):
    """Save figure to results directory."""
    out_dir = results_dir or RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.{GRAPH_FORMAT}")
    fig.savefig(path, dpi=GRAPH_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [SAVED] {path}")
    return path


# ═══════════════════════════════════════════════════════════════
#  1. LEARNING CURVE
# ═══════════════════════════════════════════════════════════════
def plot_learning_curve(
    train_acc: List[float],
    val_acc: List[float],
    early_stop_epoch: Optional[int] = None,
    results_dir: str = None,
) -> str:
    """Training vs validation accuracy over epochs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_acc) + 1)

    ax.plot(epochs, train_acc, color=COLORS["train"], linewidth=2,
            label="Training Accuracy", alpha=0.8)
    ax.plot(epochs, val_acc, color=COLORS["val"], linewidth=2,
            label="Validation Accuracy", alpha=0.8)

    if early_stop_epoch is not None:
        ax.axvline(x=early_stop_epoch, color="gray", linestyle="--",
                   alpha=0.7, label=f"Early Stop (ep {early_stop_epoch})")

    ax.axhline(y=TARGET_ACCURACY, color="green", linestyle=":",
               alpha=0.5, label=f"Target ({TARGET_ACCURACY})")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Siren AI — Learning Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    return _save_fig(fig, "01_learning_curve", results_dir)


# ═══════════════════════════════════════════════════════════════
#  2. TD ERROR / LOSS CURVE
# ═══════════════════════════════════════════════════════════════
def plot_td_error_curve(
    td_errors: List[float],
    window: int = 20,
    results_dir: str = None,
) -> str:
    """Temporal difference error over training steps."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(td_errors, color="#bdc3c7", alpha=0.3, linewidth=0.5, label="Raw TD Error")

    # Smoothed moving average
    if len(td_errors) > window:
        smoothed = pd.Series(td_errors).rolling(window=window, min_periods=1).mean()
        ax.plot(smoothed, color="#e74c3c", linewidth=2, label=f"Smoothed (w={window})")

    ax.set_xlabel("Training Episode", fontsize=12)
    ax.set_ylabel("Mean |TD Error|", fontsize=12)
    ax.set_title("Siren AI — TD Error Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    return _save_fig(fig, "02_td_error_curve", results_dir)


# ═══════════════════════════════════════════════════════════════
#  3. REWARD TREND
# ═══════════════════════════════════════════════════════════════
def plot_reward_trend(
    episode_rewards: List[float],
    window: int = 20,
    results_dir: str = None,
) -> str:
    """Episode reward trend with smoothed cumulative."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    episodes = range(len(episode_rewards))

    # Raw rewards
    ax1.plot(episodes, episode_rewards, color="#bdc3c7", alpha=0.3,
             linewidth=0.5, label="Raw")
    if len(episode_rewards) > window:
        smoothed = pd.Series(episode_rewards).rolling(window=window, min_periods=1).mean()
        ax1.plot(episodes, smoothed, color=COLORS["reward"], linewidth=2,
                 label=f"Smoothed (w={window})")
    ax1.set_ylabel("Episode Reward", fontsize=12)
    ax1.set_title("Siren AI — Reward Trend", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Cumulative reward
    cumulative = np.cumsum(episode_rewards)
    ax2.plot(episodes, cumulative, color=COLORS["reward"], linewidth=2)
    ax2.set_xlabel("Episode", fontsize=12)
    ax2.set_ylabel("Cumulative Reward", fontsize=12)
    ax2.set_title("Cumulative Reward", fontsize=12)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return _save_fig(fig, "03_reward_trend", results_dir)


# ═══════════════════════════════════════════════════════════════
#  4. POLICY ENTROPY CURVE
# ═══════════════════════════════════════════════════════════════
def plot_entropy_curve(
    entropy_values: List[float],
    entropy_floor: float = 0.30,
    collapse_events: List[int] = None,
    results_dir: str = None,
) -> str:
    """Policy entropy over training with floor and collapse markers."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(entropy_values, color=COLORS["entropy"], linewidth=2, label="Policy Entropy")
    ax.axhline(y=entropy_floor, color="red", linestyle="--", alpha=0.7,
               label=f"Entropy Floor ({entropy_floor})")

    max_entropy = np.log(NUM_MODES)
    ax.axhline(y=max_entropy, color="green", linestyle=":", alpha=0.5,
               label=f"Max Entropy ({max_entropy:.2f})")

    if collapse_events:
        for ce in collapse_events:
            ax.axvline(x=ce, color="red", alpha=0.2, linewidth=1)
        ax.axvline(x=collapse_events[0], color="red", alpha=0.3,
                   linewidth=1, label="Collapse Event")

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Policy Entropy", fontsize=12)
    ax.set_title("Siren AI — Policy Entropy Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    return _save_fig(fig, "04_entropy_curve", results_dir)


# ═══════════════════════════════════════════════════════════════
#  5. CONFUSION MATRIX
# ═══════════════════════════════════════════════════════════════
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    results_dir: str = None,
) -> str:
    """4x4 confusion matrix — normalized and raw."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_MODES)))
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=MODES, yticklabels=MODES, ax=ax1,
                vmin=0, vmax=1, cbar_kws={"label": "Proportion"})
    ax1.set_xlabel("Predicted", fontsize=12)
    ax1.set_ylabel("True", fontsize=12)
    ax1.set_title("Confusion Matrix (Normalized)", fontsize=13, fontweight="bold")

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=MODES, yticklabels=MODES, ax=ax2,
                cbar_kws={"label": "Count"})
    ax2.set_xlabel("Predicted", fontsize=12)
    ax2.set_ylabel("True", fontsize=12)
    ax2.set_title("Confusion Matrix (Raw Counts)", fontsize=13, fontweight="bold")

    fig.suptitle("Siren AI — Confusion Matrix (Test Set)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    # Also save CSV
    cm_df = pd.DataFrame(cm, index=MODES, columns=MODES)
    csv_path = os.path.join(results_dir or RESULTS_DIR, "confusion_matrix.csv")
    cm_df.to_csv(csv_path)
    print(f"  [SAVED] {csv_path}")

    return _save_fig(fig, "05_confusion_matrix", results_dir)


# ═══════════════════════════════════════════════════════════════
#  6. ACTION DISTRIBUTION
# ═══════════════════════════════════════════════════════════════
def plot_action_distribution(
    action_counts: np.ndarray,
    results_dir: str = None,
) -> str:
    """Histogram of chosen actions with entropy."""
    fig, ax = plt.subplots(figsize=(8, 6))

    total = action_counts.sum()
    probs = action_counts / max(total, 1)
    colors = [COLORS[m] for m in MODES]

    bars = ax.bar(MODES, action_counts, color=colors, edgecolor="black",
                  linewidth=0.5, alpha=0.85)

    # Add percentage labels
    for bar, count, prob in zip(bars, action_counts, probs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.01,
                f"{count:,}\n({prob:.1%})", ha="center", va="bottom", fontsize=10)

    # Compute entropy
    probs_safe = np.clip(probs, 1e-10, None)
    entropy = -np.sum(probs_safe * np.log2(probs_safe))
    max_entropy = np.log2(NUM_MODES)

    ax.set_xlabel("Action Mode", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Siren AI — Action Distribution\n"
                 f"Entropy: {entropy:.3f} / {max_entropy:.3f} (max)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    return _save_fig(fig, "06_action_distribution", results_dir)


# ═══════════════════════════════════════════════════════════════
#  7. STATE VISITATION FREQUENCY
# ═══════════════════════════════════════════════════════════════
def plot_state_visitation(
    state_visits: Dict[str, int],
    top_n: int = 20,
    results_dir: str = None,
) -> str:
    """Top N most visited states + heatmap."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Sort by frequency
    sorted_states = sorted(state_visits.items(), key=lambda x: x[1], reverse=True)
    top_states = sorted_states[:top_n]

    labels = [f"S{i}" for i in range(len(top_states))]
    counts = [c for _, c in top_states]

    # Bar chart
    ax1.barh(labels[::-1], counts[::-1], color=sns.color_palette("viridis", top_n),
             edgecolor="black", linewidth=0.3)
    ax1.set_xlabel("Visit Count", fontsize=12)
    ax1.set_ylabel("State ID", fontsize=12)
    ax1.set_title(f"Top {top_n} Most Visited States", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="x")

    # Distribution heatmap (binned)
    all_counts = np.array(list(state_visits.values()))
    if len(all_counts) > 0:
        # Create histogram bins
        n_bins = min(20, len(all_counts))
        hist, bin_edges = np.histogram(all_counts, bins=n_bins)
        heatmap_data = hist.reshape(4, -1) if len(hist) >= 4 else hist.reshape(1, -1)
        sns.heatmap(heatmap_data, cmap="YlOrRd", ax=ax2, annot=True, fmt="d",
                    cbar_kws={"label": "Frequency"})
        ax2.set_title("State Visitation Distribution", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Visit Count Bin", fontsize=11)
        ax2.set_ylabel("Bin Group", fontsize=11)

        # Imbalance detection
        if len(all_counts) > 1:
            cv = np.std(all_counts) / max(np.mean(all_counts), 1e-6)
            ax2.text(0.5, -0.15, f"CV (Imbalance): {cv:.3f}",
                     transform=ax2.transAxes, ha="center", fontsize=10,
                     color="red" if cv > 2.0 else "green")

    fig.suptitle("Siren AI — State Visitation Frequency",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    return _save_fig(fig, "07_state_visitation", results_dir)


# ═══════════════════════════════════════════════════════════════
#  8. CLASS IMBALANCE VISUALIZATION (PRE-TRAINING)
# ═══════════════════════════════════════════════════════════════
def plot_class_imbalance(
    df: pd.DataFrame,
    results_dir: str = None,
) -> str:
    """Bar charts of mode distribution, aggression ratio, breach ratio."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Mode distribution
    mode_counts = df["optimal_mode"].value_counts()
    colors = [COLORS.get(m, "#95a5a6") for m in mode_counts.index]
    mode_counts.plot(kind="bar", ax=axes[0], color=colors, edgecolor="black",
                     linewidth=0.5)
    axes[0].set_title("Mode Distribution", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Mode", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].tick_params(axis="x", rotation=0)

    # Add percentage
    total = len(df)
    for i, (mode, cnt) in enumerate(mode_counts.items()):
        axes[0].text(i, cnt + total * 0.005, f"{cnt/total:.1%}",
                     ha="center", fontsize=10)

    # Aggression ratio
    if "aggression_risk" in df.columns:
        agg_counts = df["aggression_risk"].value_counts()
        agg_colors = {"LOW": "#2ecc71", "MED": "#f39c12", "HIGH": "#e74c3c"}
        agg_counts.plot(kind="bar", ax=axes[1],
                        color=[agg_colors.get(x, "#95a5a6") for x in agg_counts.index],
                        edgecolor="black", linewidth=0.5)
        axes[1].set_title("Aggression Risk Distribution", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Risk Level", fontsize=11)
        axes[1].tick_params(axis="x", rotation=0)

    # Breach ratio
    if "boundary_breach_status" in df.columns:
        breach_counts = df["boundary_breach_status"].value_counts()
        breach_colors = {"NONE": "#2ecc71", "LIKELY": "#f39c12", "CONFIRMED": "#e74c3c"}
        breach_counts.plot(kind="bar", ax=axes[2],
                           color=[breach_colors.get(x, "#95a5a6") for x in breach_counts.index],
                           edgecolor="black", linewidth=0.5)
        axes[2].set_title("Boundary Breach Distribution", fontsize=13, fontweight="bold")
        axes[2].set_xlabel("Breach Status", fontsize=11)
        axes[2].tick_params(axis="x", rotation=0)

    fig.suptitle("Siren AI — Class Imbalance Check (Pre-Training)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    return _save_fig(fig, "08_class_imbalance", results_dir)


# ═══════════════════════════════════════════════════════════════
#  9. OVERFITTING DETECTION
# ═══════════════════════════════════════════════════════════════
def plot_overfitting_detection(
    train_acc: List[float],
    val_acc: List[float],
    results_dir: str = None,
) -> Tuple[str, bool]:
    """Plot train vs val accuracy, highlight gap, detect overfitting."""
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(train_acc) + 1)
    ax.plot(epochs, train_acc, color=COLORS["train"], linewidth=2,
            label="Training", alpha=0.8)
    ax.plot(epochs, val_acc, color=COLORS["val"], linewidth=2,
            label="Validation", alpha=0.8)

    # Fill gap region
    ax.fill_between(epochs, train_acc, val_acc,
                    alpha=0.15, color="red", label="Gap Region")

    # Detect overfitting
    overfit = False
    if len(train_acc) > 0 and len(val_acc) > 0:
        final_train = train_acc[-1]
        final_val = val_acc[-1]
        gap = final_train - final_val

        if final_train > OVERFIT_TRAIN_THRESHOLD and final_val < OVERFIT_VAL_THRESHOLD:
            overfit = True
        if gap > OVERFIT_GAP_THRESHOLD:
            overfit = True

        status = "⚠ OVERFITTING DETECTED" if overfit else "✓ No Overfitting"
        color = "red" if overfit else "green"
        ax.text(0.02, 0.02, f"{status}\nGap: {gap:.3f}  Train: {final_train:.3f}  Val: {final_val:.3f}",
                transform=ax.transAxes, fontsize=11, color=color,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Siren AI — Overfitting Detection", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    path = _save_fig(fig, "09_overfitting_detection", results_dir)
    return path, overfit


# ═══════════════════════════════════════════════════════════════
#  10. Q-VALUE STABILITY
# ═══════════════════════════════════════════════════════════════
def plot_q_value_stability(
    q_magnitudes: List[float],
    results_dir: str = None,
) -> str:
    """Average Q-value magnitude over time."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(q_magnitudes, color=COLORS["q_value"], linewidth=2,
            label="|Q| Average")

    # Detect issues
    if len(q_magnitudes) > 10:
        max_q = max(q_magnitudes)
        min_q = min(q_magnitudes[-len(q_magnitudes)//4:])

        if max_q > 8:
            ax.axhline(y=max_q, color="red", linestyle="--", alpha=0.5,
                       label=f"Max |Q| = {max_q:.2f} (EXPLOSION?)")
        if min_q < 0.01:
            ax.text(0.98, 0.02, "⚠ Q-values near zero (COLLAPSE?)",
                    transform=ax.transAxes, ha="right", fontsize=10, color="red")

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Average |Q| Magnitude", fontsize=12)
    ax.set_title("Siren AI — Q-Value Stability", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    return _save_fig(fig, "10_q_value_stability", results_dir)


# ═══════════════════════════════════════════════════════════════
#  11. MULTI-SEED STABILITY
# ═══════════════════════════════════════════════════════════════
def plot_multi_seed_stability(
    seed_accuracies: Dict[int, float],
    seed_histories: Dict[int, List[float]] = None,
    results_dir: str = None,
) -> str:
    """Accuracy per seed with variance analysis."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    seeds = list(seed_accuracies.keys())
    accs = list(seed_accuracies.values())

    # Bar chart
    colors_bar = sns.color_palette("husl", len(seeds))
    ax1.bar([f"Seed {s}" for s in seeds], accs, color=colors_bar,
            edgecolor="black", linewidth=0.5)
    ax1.axhline(y=np.mean(accs), color="red", linestyle="--",
                label=f"Mean: {np.mean(accs):.4f}")
    ax1.axhline(y=TARGET_ACCURACY, color="green", linestyle=":",
                label=f"Target: {TARGET_ACCURACY}")

    variance = np.var(accs)
    std = np.std(accs)
    ax1.set_title(f"Accuracy per Seed\nVariance: {variance:.6f}, Std: {std:.4f}",
                  fontsize=13, fontweight="bold")
    ax1.set_ylabel("Test Accuracy", fontsize=12)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis="y")

    # Learning curves per seed
    if seed_histories:
        for seed, hist in seed_histories.items():
            ax2.plot(hist, linewidth=1.5, alpha=0.7, label=f"Seed {seed}")
        ax2.set_xlabel("Episode", fontsize=12)
        ax2.set_ylabel("Episode Accuracy", fontsize=12)
        ax2.set_title("Learning Curves per Seed", fontsize=13, fontweight="bold")
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No per-seed histories available",
                 transform=ax2.transAxes, ha="center", fontsize=12)

    fig.suptitle("Siren AI — Multi-Seed Stability",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    return _save_fig(fig, "11_multi_seed_stability", results_dir)


# ═══════════════════════════════════════════════════════════════
#  METRICS COMPUTATION
# ═══════════════════════════════════════════════════════════════
def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_name: str = "test",
) -> dict:
    """Compute all required metrics."""
    metrics = {}

    # Accuracy
    metrics[f"{split_name}_accuracy"] = float(accuracy_score(y_true, y_pred))

    # Per-mode precision, recall, F1
    for avg in ["macro", "weighted"]:
        metrics[f"{split_name}_precision_{avg}"] = float(
            precision_score(y_true, y_pred, average=avg, zero_division=0)
        )
        metrics[f"{split_name}_recall_{avg}"] = float(
            recall_score(y_true, y_pred, average=avg, zero_division=0)
        )
        metrics[f"{split_name}_f1_{avg}"] = float(
            f1_score(y_true, y_pred, average=avg, zero_division=0)
        )

    # Per-mode metrics
    per_mode_precision = precision_score(y_true, y_pred, average=None,
                                         labels=list(range(NUM_MODES)),
                                         zero_division=0)
    per_mode_recall = recall_score(y_true, y_pred, average=None,
                                   labels=list(range(NUM_MODES)),
                                   zero_division=0)
    per_mode_f1 = f1_score(y_true, y_pred, average=None,
                           labels=list(range(NUM_MODES)),
                           zero_division=0)

    for i, mode in enumerate(MODES):
        metrics[f"{split_name}_precision_{mode}"] = float(per_mode_precision[i])
        metrics[f"{split_name}_recall_{mode}"] = float(per_mode_recall[i])
        metrics[f"{split_name}_f1_{mode}"] = float(per_mode_f1[i])

    # v3: Under-escalation rate — fraction of samples where predicted < optimal
    under_escalation_count = int(np.sum(y_pred < y_true))
    under_escalation_rate = under_escalation_count / len(y_true) if len(y_true) > 0 else 0.0
    metrics[f"{split_name}_under_escalation_rate"] = float(under_escalation_rate)

    return metrics


def print_metrics(metrics: dict, split_name: str = "test"):
    """Print metrics in a clear format."""
    print(f"\n{'='*60}")
    print(f"  SIREN AI — FINAL METRICS ({split_name.upper()})")
    print(f"{'='*60}")

    acc = metrics.get(f"{split_name}_accuracy", 0)
    print(f"\n  Final {split_name.title()} Accuracy: {acc:.4f}")

    if acc < TARGET_ACCURACY:
        print(f"  ⚠ WARNING: Accuracy {acc:.4f} < target {TARGET_ACCURACY}")
        print(f"  ❌ MODEL MARKED AS FAILED")

    print(f"\n  Macro F1:    {metrics.get(f'{split_name}_f1_macro', 0):.4f}")
    print(f"  Weighted F1: {metrics.get(f'{split_name}_f1_weighted', 0):.4f}")

    uer = metrics.get(f"{split_name}_under_escalation_rate", 0)
    uer_status = "✅" if uer <= 0.12 else "❌"
    print(f"  Under-Escalation Rate: {uer:.4f}  {uer_status} (target ≤ 0.12)")

    print(f"\n  Per-Mode Metrics:")
    print(f"  {'Mode':<6} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print(f"  {'─'*42}")
    for mode in MODES:
        p = metrics.get(f"{split_name}_precision_{mode}", 0)
        r = metrics.get(f"{split_name}_recall_{mode}", 0)
        f = metrics.get(f"{split_name}_f1_{mode}", 0)
        print(f"  {mode:<6} {p:<12.4f} {r:<12.4f} {f:<12.4f}")

    print(f"\n{'='*60}\n")


def check_failure_conditions(
    metrics: dict,
    entropy_final: float,
    entropy_threshold: float,
    overfit_detected: bool,
    q_explosion: bool,
    action_collapse: bool,
    class_imbalance: bool,
) -> Tuple[bool, List[str]]:
    """
    Check all failure conditions.
    Returns (model_rejected, list_of_reasons).
    """
    reasons = []

    acc = metrics.get("test_accuracy", 0)
    if acc < TARGET_ACCURACY:
        reasons.append(f"Accuracy {acc:.4f} < {TARGET_ACCURACY}")

    if entropy_final < entropy_threshold:
        reasons.append(f"Entropy {entropy_final:.4f} < {entropy_threshold}")

    if class_imbalance:
        reasons.append("Severe class imbalance detected")

    if overfit_detected:
        reasons.append("Overfitting gap > 20%")

    if q_explosion:
        reasons.append("Q-value explosion detected")

    if action_collapse:
        reasons.append("Deterministic action collapse")

    # v3: Under-escalation hard rejection
    uer = metrics.get("test_under_escalation_rate", 0)
    if uer > 0.12:
        reasons.append(f"Under-escalation rate {uer:.4f} > 0.12")

    rejected = len(reasons) > 0
    return rejected, reasons


# ═══════════════════════════════════════════════════════════════
#  EXPORT LOGS
# ═══════════════════════════════════════════════════════════════
def export_all_logs(
    metrics: dict,
    entropy_log: List[float],
    reward_log: List[float],
    results_dir: str = None,
):
    """Export all logs as JSON/CSV."""
    out_dir = results_dir or RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    # metrics.json
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  [SAVED] {os.path.join(out_dir, 'metrics.json')}")

    # entropy_log.csv
    pd.DataFrame({"epoch": range(len(entropy_log)), "entropy": entropy_log}).to_csv(
        os.path.join(out_dir, "entropy_log.csv"), index=False
    )
    print(f"  [SAVED] {os.path.join(out_dir, 'entropy_log.csv')}")

    # reward_log.csv
    pd.DataFrame({"episode": range(len(reward_log)), "reward": reward_log}).to_csv(
        os.path.join(out_dir, "reward_log.csv"), index=False
    )
    print(f"  [SAVED] {os.path.join(out_dir, 'reward_log.csv')}")
