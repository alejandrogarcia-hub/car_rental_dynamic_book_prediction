"""
Compare temporal distributions between original, standard SDV, and improved SDV synthetic data.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_and_analyze_searches(filepath, label):
    """Load search data and extract temporal distributions."""
    df = pd.read_csv(filepath)
    df["search_ts"] = pd.to_datetime(df["search_ts"])

    # Extract distributions
    hour_dist = df["search_ts"].dt.hour.value_counts(normalize=True).sort_index()
    dow_dist = df["search_ts"].dt.dayofweek.value_counts(normalize=True).sort_index()
    month_dist = df["search_ts"].dt.month.value_counts(normalize=True).sort_index()

    # Key metrics
    midnight_pct = hour_dist.get(0, 0)
    peak_hour = hour_dist.idxmax()
    peak_hour_pct = hour_dist.max()

    return {
        "label": label,
        "hour_dist": hour_dist,
        "dow_dist": dow_dist,
        "month_dist": month_dist,
        "midnight_pct": midnight_pct,
        "peak_hour": peak_hour,
        "peak_hour_pct": peak_hour_pct,
        "total_searches": len(df),
    }


def plot_comparison(datasets):
    """Create comparison plots for temporal distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Temporal Distribution Comparison: Original vs SDV Methods", fontsize=16
    )

    colors = ["steelblue", "darkorange", "darkgreen"]
    markers = ["o", "s", "^"]

    # 1. Hourly distribution comparison
    ax = axes[0, 0]
    for i, data in enumerate(datasets):
        hours = range(24)
        values = [data["hour_dist"].get(h, 0) for h in hours]
        ax.plot(
            hours,
            values,
            f"{markers[i]}-",
            label=data["label"],
            color=colors[i],
            linewidth=2,
            markersize=6,
            alpha=0.8,
        )

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Proportion of Searches")
    ax.set_title("Hourly Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 24, 3))

    # 2. Focus on problematic hours (midnight issue)
    ax = axes[0, 1]
    problem_hours = [0, 1, 2, 22, 23]  # Late night/early morning
    x = np.arange(len(problem_hours))
    width = 0.25

    for i, data in enumerate(datasets):
        values = [data["hour_dist"].get(h, 0) * 100 for h in problem_hours]
        ax.bar(
            x + i * width,
            values,
            width,
            label=data["label"],
            color=colors[i],
            alpha=0.8,
        )

    ax.set_xlabel("Hour")
    ax.set_ylabel("Percentage of Searches")
    ax.set_title("Late Night/Early Morning Hours (Midnight Issue)")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{h}:00" for h in problem_hours])
    ax.legend()

    # 3. Day of week comparison
    ax = axes[1, 0]
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    x = np.arange(7)

    for i, data in enumerate(datasets):
        values = [data["dow_dist"].get(d, 0) * 100 for d in range(7)]
        ax.plot(
            x,
            values,
            f"{markers[i]}-",
            label=data["label"],
            color=colors[i],
            linewidth=2,
            markersize=8,
            alpha=0.8,
        )

    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Percentage of Searches")
    ax.set_title("Day of Week Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(days)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Quality metrics comparison
    ax = axes[1, 1]
    metrics = ["Midnight %", "Peak Hour %", "Total Searches"]

    # Prepare data
    metric_data = [
        [
            data["midnight_pct"] * 100,
            data["peak_hour_pct"] * 100,
            data["total_searches"] / 1000,  # Scale for visualization
        ]
        for data in datasets
    ]

    x = np.arange(len(metrics))
    width = 0.25

    for i, (data, values) in enumerate(zip(datasets, metric_data)):
        ax.bar(
            x + i * width,
            values,
            width,
            label=data["label"],
            color=colors[i],
            alpha=0.8,
        )

    ax.set_ylabel("Value")
    ax.set_title("Key Metrics Comparison")
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.legend()

    # Add value labels
    for i, values in enumerate(metric_data):
        for j, v in enumerate(values):
            if j == 2:  # Total searches
                label = f"{v:.1f}k"
            else:
                label = f"{v:.1f}%"
            ax.text(j + i * width, v + 0.5, label, ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    return fig


def calculate_distribution_quality(original_data, synthetic_data):
    """Calculate quality metrics for distribution matching."""
    # Hourly distribution difference
    hour_diffs = []
    for h in range(24):
        orig_val = original_data["hour_dist"].get(h, 0)
        synth_val = synthetic_data["hour_dist"].get(h, 0)
        hour_diffs.append(abs(orig_val - synth_val))

    mean_hour_diff = np.mean(hour_diffs)
    max_hour_diff = np.max(hour_diffs)

    # Specific metrics
    midnight_diff = abs(original_data["midnight_pct"] - synthetic_data["midnight_pct"])

    # Day of week difference
    dow_diffs = []
    for d in range(7):
        orig_val = original_data["dow_dist"].get(d, 0)
        synth_val = synthetic_data["dow_dist"].get(d, 0)
        dow_diffs.append(abs(orig_val - synth_val))

    mean_dow_diff = np.mean(dow_diffs)

    return {
        "mean_hour_diff": mean_hour_diff,
        "max_hour_diff": max_hour_diff,
        "midnight_diff": midnight_diff,
        "mean_dow_diff": mean_dow_diff,
        "overall_quality": 1 - np.mean([mean_hour_diff, mean_dow_diff]),
    }


def main():
    """Run the comparison analysis."""
    # Define file paths
    base_path = Path("../../../data")

    # Load datasets
    print("Loading search data...")
    datasets = []

    # Original data
    original_path = base_path / "sample" / "searches.csv"
    if original_path.exists():
        datasets.append(load_and_analyze_searches(original_path, "Original"))
        print("✓ Loaded original data")

    # Standard SDV synthetic
    sdv_path = base_path / "synthetic_data" / "synthetic_searches.csv"
    if sdv_path.exists():
        datasets.append(load_and_analyze_searches(sdv_path, "Standard SDV"))
        print("✓ Loaded standard SDV data")

    # Improved SDV synthetic
    improved_path = base_path / "improved_synthetic_data" / "synthetic_searches.csv"
    if improved_path.exists():
        datasets.append(load_and_analyze_searches(improved_path, "Improved SDV"))
        print("✓ Loaded improved SDV data")

    if len(datasets) < 2:
        print("Error: Need at least original and one synthetic dataset for comparison")
        return

    # Create comparison plots
    print("\nCreating comparison plots...")
    fig = plot_comparison(datasets)

    # Save plot
    output_path = "temporal_distribution_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved plot to {output_path}")

    # Calculate and print quality metrics
    print("\n" + "=" * 60)
    print("DISTRIBUTION QUALITY METRICS")
    print("=" * 60)

    original = datasets[0]
    for synthetic in datasets[1:]:
        quality = calculate_distribution_quality(original, synthetic)

        print(f"\n{synthetic['label']}:")
        print(
            f"  Mean hourly difference: {quality['mean_hour_diff']:.4f} ({quality['mean_hour_diff'] * 100:.2f}%)"
        )
        print(
            f"  Max hourly difference: {quality['max_hour_diff']:.4f} ({quality['max_hour_diff'] * 100:.2f}%)"
        )
        print(
            f"  Midnight hour difference: {quality['midnight_diff']:.4f} ({quality['midnight_diff'] * 100:.2f}%)"
        )
        print(
            f"  Mean day-of-week difference: {quality['mean_dow_diff']:.4f} ({quality['mean_dow_diff'] * 100:.2f}%)"
        )
        print(
            f"  Overall temporal quality: {quality['overall_quality']:.4f} ({quality['overall_quality'] * 100:.1f}%)"
        )

        # Quality assessment
        if quality["mean_hour_diff"] < 0.02:
            print("  Assessment: ✅ EXCELLENT - Temporal patterns well preserved")
        elif quality["mean_hour_diff"] < 0.05:
            print("  Assessment: ✓ GOOD - Acceptable temporal preservation")
        else:
            print("  Assessment: ⚠️  POOR - Significant temporal distribution loss")

    # Show specific improvements
    if len(datasets) == 3:  # Have both standard and improved
        standard_quality = calculate_distribution_quality(original, datasets[1])
        improved_quality = calculate_distribution_quality(original, datasets[2])

        print("\n" + "=" * 60)
        print("IMPROVEMENT SUMMARY")
        print("=" * 60)

        improvements = {
            "Hourly distribution": (
                standard_quality["mean_hour_diff"] - improved_quality["mean_hour_diff"]
            )
            / standard_quality["mean_hour_diff"]
            * 100,
            "Midnight hour": (
                standard_quality["midnight_diff"] - improved_quality["midnight_diff"]
            )
            / standard_quality["midnight_diff"]
            * 100,
            "Day of week": (
                standard_quality["mean_dow_diff"] - improved_quality["mean_dow_diff"]
            )
            / standard_quality["mean_dow_diff"]
            * 100,
        }

        for metric, improvement in improvements.items():
            if improvement > 0:
                print(f"  {metric}: {improvement:.1f}% improvement ✅")
            else:
                print(f"  {metric}: {abs(improvement):.1f}% worse ❌")

    plt.show()


if __name__ == "__main__":
    main()
