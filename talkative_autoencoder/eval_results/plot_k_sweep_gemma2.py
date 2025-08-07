#!/usr/bin/env python3
"""
Plot variance recovery vs K for best-of-K sweep results.

Usage:
    # Plot all gemma2 JSON files in eval_results/
    source scripts/ensure_env.sh && uv run python eval_results/plot_k_sweep.py

    # Plot specific files
    source scripts/ensure_env.sh && uv run python eval_results/plot_k_sweep.py file1.json file2.json

    # Plot with custom output name
    source scripts/ensure_env.sh && uv run python eval_results/plot_k_sweep.py -o my_plot.png file1.json file2.json

Note: This script requires matplotlib, which is available in the consistency-lens environment.
Always run with 'source scripts/ensure_env.sh && uv run python' to ensure the correct environment.

The script will:
- Find all gemma2*.json files if no files are specified
- Create a single plot with all runs on the same graph
- X-axis: Best of N (K values) on log scale
- Y-axis: Variance Recovered (auto-scaled to data range)
- Save as both PNG and PDF formats
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _plot_histograms(result_file, data, bins=50):
    """Create MSE (per-k) and activation-norm histograms."""
    results = data["results_by_k"]

    # MSE histograms ─ one per k
    for r in results:
        if "mse_values" not in r:
            continue
        mses = np.asarray(r["mse_values"])
        plt.figure(figsize=(6, 4))
        plt.hist(mses, bins=bins, color="steelblue", alpha=0.8)
        plt.xlabel("MSE")
        plt.ylabel("Count")
        plt.title(f"MSE distribution (K={r['k']})")
        out_path = Path(result_file).parent / f"plots/{Path(result_file).stem}_mse_hist_k{r['k']}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

    # Activation-norm histogram (same for all k)
    if results and "activation_norms" in results[0]:
        norms = np.asarray(results[0]["activation_norms"])
        plt.figure(figsize=(6, 4))
        plt.hist(norms, bins=bins, color="darkorange", alpha=0.8)
        plt.xlabel("‖A‖")
        plt.ylabel("Count")
        plt.title("Activation vector norms")
        out_path = Path(result_file).parent / f"plots/{Path(result_file).stem}_activation_norms_hist.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()


def load_results(json_file):
    """Load results from JSON file, handling potential formatting issues."""
    with open(json_file, "r") as f:
        content = f.read()

    # Try to parse as-is first
    try:
        data = json.loads(content)
        return data
    except json.JSONDecodeError as e:
        print(f"Warning: JSON decode error in {json_file}: {e}")
        # Try to fix common issues like trailing commas
        # Remove trailing commas before closing braces/brackets
        import re

        fixed_content = re.sub(r",\s*([\]}])", r"\1", content)
        try:
            data = json.loads(fixed_content)
            print(f"Successfully parsed {json_file} after fixing trailing commas")
            return data
        except:
            # If still fails, try to extract just the results_by_k part
            print(f"Failed to parse {json_file}, skipping...")
            raise


def _plot_combined_mse_histograms(all_data, output_file, bins=50):
    """Plot all MSE histograms on a single plot."""
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(12, 9))
    plt.style.use("seaborn-v0_8-darkgrid")

    # Count total number of histograms
    total_hists = sum(len(data["results_by_k"]) for _, data in all_data if "results_by_k" in data)
    if total_hists == 0:
        return

    colors = plt.cm.viridis(np.linspace(0, 1, total_hists))
    current_hist = 0

    for result_file, data in all_data:
        results = data["results_by_k"]
        label_prefix = Path(result_file).stem.replace("gemma2", "").replace("_", " ").strip()
        for r in results:
            if "mse_values" not in r:
                continue
            mses = np.asarray(r["mse_values"])
            label = f"{label_prefix} K={r['k']}"
            mses = mses[(mses >= 0) & (mses <= 100000)]
            plt.hist(
                mses,
                bins=bins,
                alpha=0.7,
                label=label,
                color=colors[current_hist],
                histtype="stepfilled",
                linewidth=1.5,
            )
            current_hist += 1

    plt.xlabel("MSE", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.title("Combined MSE Distributions", fontsize=16)
    plt.grid(True, alpha=0.3)

    if total_hists > 1:
        plt.legend(loc="upper right", fontsize=10)

    output_dir = Path(__file__).parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(output_file).stem}_mse_histograms.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Combined MSE histogram plot saved to {output_path}")


def _plot_per_file_mse_histograms(result_file, data, output_dir, bins=50):
    """Plot all MSE histograms for a single file on one plot."""

    import matplotlib.pyplot as plt
    import numpy as np

    results = data["results_by_k"]
    if not results:
        return

    plt.figure(figsize=(8, 6))
    plt.style.use("seaborn-v0_8-darkgrid")
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    any_hist = False

    for idx, r in enumerate(results):
        if "mse_values" not in r:
            continue
        mses = np.asarray(r["mse_values"])
        label = f"K={r['k']}"
        mses = mses[(mses >= 0) & (mses <= 100000)]
        plt.hist(
            mses,
            bins=bins,
            alpha=0.7,
            label=label,
            color=colors[idx],
            histtype="stepfilled",
            linewidth=1.5,
        )
        any_hist = True

    if not any_hist:
        plt.close()
        return

    plt.xlabel("MSE", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.title("MSE Distributions", fontsize=16)
    plt.grid(True, alpha=0.3)
    if len(results) > 1:
        plt.legend(loc="upper right", fontsize=10)

    output_path = output_dir / "mse_histograms.png"
    plt.tight_layout()
    plt.yscale("log")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"MSE histogram plot saved to {output_path}")


def plot_token_statistics(result_file, data, output_dir):
    """Plot token statistics showing proportions of selected tokens."""
    # Get the first result that has selected_token_stats
    selected_stats = None
    for r in data["results_by_k"]:
        if "selected_token_stats" in r:
            selected_stats = r["selected_token_stats"]
            break

    if not selected_stats or "proportions" not in selected_stats:
        return

    proportions = selected_stats["proportions"]

    # Create pie chart
    labels = []
    sizes = []
    colors = []
    color_map = {
        "user": "#0072B2",
        "assistant": "#D55E00",
        "other": "#009E73",
        "special": "#CC79A7",
        "system": "#F0E442",
        "pad": "#999999",
    }

    for role, proportion in proportions.items():
        if proportion > 0 and role != "total":
            labels.append(f"{role.capitalize()} ({proportion:.1%})")
            sizes.append(proportion)
            colors.append(color_map.get(role, "#000000"))

    if not sizes:
        return

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    plt.title("Distribution of Selected Token Types", fontsize=16)
    plt.axis("equal")

    output_path = output_dir / "selected_token_distribution.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Token distribution plot saved to {output_path}")


def plot_mse_by_role(result_file, data, output_dir):
    """Plot MSE by token role (user vs assistant vs other) for each K value."""
    results = data["results_by_k"]

    # Extract K values and MSE by role data
    k_values = []
    user_mses = []
    assistant_mses = []
    other_mses = []
    special_mses = []

    for r in results:
        if "mse_by_role" not in r:
            continue

        k_values.append(r["k"])
        mse_by_role = r["mse_by_role"]

        # Extract MSE values, defaulting to None if not present
        user_mses.append(mse_by_role.get("user", {}).get("avg_mse", None))
        assistant_mses.append(mse_by_role.get("assistant", {}).get("avg_mse", None))
        other_mses.append(mse_by_role.get("other", {}).get("avg_mse", None))
        special_mses.append(mse_by_role.get("special", {}).get("avg_mse", None))

    if not k_values:
        return

    plt.figure(figsize=(10, 6))
    plt.style.use("seaborn-v0_8-darkgrid")

    # Plot lines for each role
    if any(m is not None for m in user_mses):
        plt.plot(k_values, user_mses, "o-", color="#0072B2", linewidth=2.5, markersize=8, label="User tokens")
    if any(m is not None for m in assistant_mses):
        plt.plot(k_values, assistant_mses, "s-", color="#D55E00", linewidth=2.5, markersize=8, label="Assistant tokens")
    if any(m is not None for m in other_mses):
        plt.plot(k_values, other_mses, "^-", color="#009E73", linewidth=2.5, markersize=8, label="Other tokens")
    if any(m is not None for m in special_mses):
        plt.plot(k_values, special_mses, "v-", color="#CC79A7", linewidth=2.5, markersize=8, label="Special tokens")

    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Best of N", fontsize=14)
    plt.ylabel("Average MSE", fontsize=14)
    plt.title("MSE by Token Role", fontsize=16)
    plt.xticks(k_values, [str(k) for k in k_values])
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=12)
    plt.tight_layout()

    output_path = output_dir / "mse_by_role.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"MSE by role plot saved to {output_path}")


def plot_variance_recovery_by_role(result_file, data, output_dir):
    """Plot variance recovery by token role (user vs assistant vs other) for each K value."""
    results = data["results_by_k"]

    # Extract K values and variance recovery by role data
    k_values = []
    user_vr = []
    assistant_vr = []
    other_vr = []
    special_vr = []

    for r in results:
        if "variance_recovery_by_role" not in r:
            continue

        k_values.append(r["k"])
        vr_by_role = r["variance_recovery_by_role"]

        # Extract variance recovery values, defaulting to None if not present
        user_vr.append(vr_by_role.get("user", {}).get("variance_recovery", None))
        assistant_vr.append(vr_by_role.get("assistant", {}).get("variance_recovery", None))
        other_vr.append(vr_by_role.get("other", {}).get("variance_recovery", None))
        special_vr.append(vr_by_role.get("special", {}).get("variance_recovery", None))

    if not k_values:
        return

    plt.figure(figsize=(10, 6))
    plt.style.use("seaborn-v0_8-darkgrid")

    # Plot lines for each role
    if any(v is not None for v in user_vr):
        plt.plot(k_values, user_vr, "o-", color="#0072B2", linewidth=2.5, markersize=8, label="User tokens")
    if any(v is not None for v in assistant_vr):
        plt.plot(k_values, assistant_vr, "s-", color="#D55E00", linewidth=2.5, markersize=8, label="Assistant tokens")
    if any(v is not None for v in other_vr):
        plt.plot(k_values, other_vr, "^-", color="#009E73", linewidth=2.5, markersize=8, label="Other tokens")
    if any(v is not None for v in special_vr):
        plt.plot(k_values, special_vr, "v-", color="#CC79A7", linewidth=2.5, markersize=8, label="Special tokens")

    plt.xscale("log", base=2)
    plt.xlabel("Best of N", fontsize=14)
    plt.ylabel("Variance Recovery", fontsize=14)
    plt.title("Variance Recovery by Token Role", fontsize=16)
    plt.xticks(k_values, [str(k) for k in k_values])
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=12)
    plt.tight_layout()

    output_path = output_dir / "variance_recovery_by_role.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Variance recovery by role plot saved to {output_path}")


def plot_variance_recovery(results_files, output_file="variance_recovery_plot.png", plot_histograms=False):
    """Plot variance recovery vs K for multiple result files."""

    plt.figure(figsize=(10, 8))

    # Set style
    plt.style.use("seaborn-v0_8-darkgrid")

    # Colorblind-friendly palette (Color Universal Design - CUD)
    # See: https://jfly.uni-koeln.de/color/
    colorblind_palette = [
        "#000000",  # black
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
        "#009E73",  # bluish green
        "#F0E442",  # yellow
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#CC79A7",  # reddish purple
    ]
    # Repeat palette if more files than colors
    colors = [colorblind_palette[i % len(colorblind_palette)] for i in range(len(results_files))]

    valid_files = []
    all_data = []

    # Load all files first, skipping any that fail
    for result_file in results_files:
        try:
            data = load_results(result_file)
            all_data.append((result_file, data))
            valid_files.append(result_file)
        except Exception as e:
            print(f"Skipping {result_file} due to error: {e}")
            continue

    if not valid_files:
        print("No valid result files found!")
        return

    for idx, (result_file, data) in enumerate(all_data):
        # Extract K values and variance recovery
        results = data["results_by_k"]
        k_values = [r["k"] for r in results]
        var_recovery = [r["variance_recovery"] for r in results]

        # Skip confidence intervals
        has_ci = False

        # Extract label from filename
        label = Path(result_file).stem.replace("gemma2", "").replace("_", " ").strip()
        if not label:
            label = f"Run {idx + 1}"

        # Plot line
        plt.plot(k_values, var_recovery, "o-", color=colors[idx], linewidth=2.5, markersize=8, label=label)

        # No confidence intervals
        # Histograms
        # _plot_histograms(result_file, data)

    # Set log scale for x-axis
    plt.xscale("log", base=2)

    # Labels and title
    plt.xlabel("Best of N", fontsize=14)
    plt.ylabel("Variance Recovered", fontsize=14)
    plt.title("Gemma2 9B Test Time Interpretability", fontsize=16, pad=20)

    # Set x-axis ticks to show actual K values
    # Extract unique k values from already loaded data
    unique_k = sorted(set(k for _, data in all_data for k in [r["k"] for r in data["results_by_k"]]))
    plt.xticks(unique_k, [str(k) for k in unique_k])

    # Grid
    plt.grid(True, alpha=0.3)

    # Legend
    if len(results_files) > 1:
        plt.legend(loc="lower right", fontsize=12)

    # Y-axis limits - auto-scale to data
    # Get min and max values from all data
    all_var_recovery = []
    for _, data in all_data:
        results = data["results_by_k"]
        all_var_recovery.extend([r["variance_recovery"] for r in results])

    if all_var_recovery:
        y_min = min(all_var_recovery) * 0.95  # Add 5% padding
        y_max = max(all_var_recovery) * 1.05  # Add 5% padding
        plt.ylim(y_min, y_max)

    # Tight layout
    plt.tight_layout()

    # Save figure
    output_dir = Path(__file__).parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / output_file
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_file}")

    # # Also save as PDF for publication quality
    # plt.savefig(pdf_file, bbox_inches="tight")
    # print(f"PDF version saved to {pdf_file}")

    plt.close()

    if plot_histograms:
        # Create individual plots for each file in its own folder
        for result_file, data in all_data:
            # Create folder for this specific JSON file
            file_stem = Path(result_file).stem
            individual_output_dir = output_dir / file_stem
            individual_output_dir.mkdir(parents=True, exist_ok=True)

            # Plot MSE histograms
            _plot_per_file_mse_histograms(result_file, data, individual_output_dir)

            # Plot MSE by role
            plot_mse_by_role(result_file, data, individual_output_dir)

            # Plot variance recovery by role
            plot_variance_recovery_by_role(result_file, data, individual_output_dir)

            # Plot activation norm histogram
            if data["results_by_k"] and "activation_norms" in data["results_by_k"][0]:
                norms = np.asarray(data["results_by_k"][0]["activation_norms"])
                plt.figure(figsize=(6, 4))
                plt.hist(norms, bins=50, color="darkorange", alpha=0.8)
                plt.xlabel("‖A‖")
                plt.ylabel("Count")
                plt.yscale("log")
                plt.title("Activation Vector Norms", fontsize=16)
                out_path = individual_output_dir / "activation_norms_hist.png"
                plt.tight_layout()
                plt.savefig(out_path, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"Activation norms histogram saved to {out_path}")

            # Plot token statistics if available
            if data["results_by_k"] and "selected_token_stats" in data["results_by_k"][0]:
                plot_token_statistics(result_file, data, individual_output_dir)


def plot_combined_mse_by_role(all_data, output_dir):
    """Create a combined plot showing MSE by role for all files."""
    # Check if any file has mse_by_role data
    has_mse_by_role = False
    for _, data in all_data:
        for r in data.get("results_by_k", []):
            if "mse_by_role" in r:
                has_mse_by_role = True
                break
        if has_mse_by_role:
            break

    if not has_mse_by_role:
        print("No mse_by_role data found in any files. Skipping combined MSE by role plot.")
        return

    plt.figure(figsize=(12, 8))
    plt.style.use("seaborn-v0_8-darkgrid")

    # Use different line styles for different files
    line_styles = ["-", "--", "-.", ":"]
    marker_styles = ["o", "s", "^", "v", "D", "p", "*", "h"]

    plot_has_data = False

    for file_idx, (result_file, data) in enumerate(all_data):
        results = data.get("results_by_k", [])

        # Extract K values and MSE by role data
        k_values = []
        user_mses = []
        assistant_mses = []

        for r in results:
            if "mse_by_role" not in r:
                continue

            k_values.append(r["k"])
            mse_by_role = r["mse_by_role"]

            # Focus on user vs assistant
            user_mse = mse_by_role.get("user", {}).get("avg_mse", None)
            assistant_mse = mse_by_role.get("assistant", {}).get("avg_mse", None)

            user_mses.append(user_mse)
            assistant_mses.append(assistant_mse)

        if not k_values:
            continue

        # Extract label from filename
        label_prefix = (
            Path(result_file).stem.replace("gemma2", "").replace("k_sweep_results_", "").replace("_", " ").strip()
        )
        if not label_prefix:
            label_prefix = f"Run {file_idx + 1}"

        line_style = line_styles[file_idx % len(line_styles)]
        marker_style = marker_styles[file_idx % len(marker_styles)]

        # Filter out None values and plot
        user_data = [(k, m) for k, m in zip(k_values, user_mses) if m is not None]
        if user_data:
            k_vals, mse_vals = zip(*user_data)
            plt.plot(
                k_vals,
                mse_vals,
                marker=marker_style,
                linestyle=line_style,
                color="#0072B2",
                linewidth=2.5,
                markersize=8,
                label=f"{label_prefix} - User",
                alpha=0.8,
            )
            plot_has_data = True

        assistant_data = [(k, m) for k, m in zip(k_values, assistant_mses) if m is not None]
        if assistant_data:
            k_vals, mse_vals = zip(*assistant_data)
            plt.plot(
                k_vals,
                mse_vals,
                marker=marker_style,
                linestyle=line_style,
                color="#D55E00",
                linewidth=2.5,
                markersize=8,
                label=f"{label_prefix} - Assistant",
                alpha=0.8,
            )
            plot_has_data = True

    if not plot_has_data:
        plt.close()
        print("No valid MSE by role data to plot.")
        return

    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Best of N", fontsize=14)
    plt.ylabel("Average MSE", fontsize=14)
    plt.title("MSE by Token Role - All Runs", fontsize=16)

    # Get unique k values for x-ticks
    unique_k = sorted(
        set(k for _, data in all_data for r in data.get("results_by_k", []) if "mse_by_role" in r for k in [r["k"]])
    )
    if unique_k:
        plt.xticks(unique_k, [str(k) for k in unique_k])

    plt.grid(True, alpha=0.3)

    # Only add legend if we have data
    if plot_has_data:
        plt.legend(loc="upper right", fontsize=10, ncol=2)

    plt.tight_layout()

    output_path = output_dir / "gemma2_combined_mse_by_role.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Combined MSE by role plot saved to {output_path}")


def plot_combined_variance_recovery_by_role(all_data, output_dir):
    """Create a combined plot showing variance recovery by role for all files."""
    # Check if any file has variance_recovery_by_role data
    has_vr_by_role = False
    for _, data in all_data:
        for r in data.get("results_by_k", []):
            if "variance_recovery_by_role" in r:
                has_vr_by_role = True
                break
        if has_vr_by_role:
            break

    if not has_vr_by_role:
        print("No variance_recovery_by_role data found in any files. Skipping combined variance recovery by role plot.")
        return

    plt.figure(figsize=(12, 8))
    plt.style.use("seaborn-v0_8-darkgrid")

    # Use different line styles for different files
    line_styles = ["-", "--", "-.", ":"]
    marker_styles = ["o", "s", "^", "v", "D", "p", "*", "h"]

    plot_has_data = False

    for file_idx, (result_file, data) in enumerate(all_data):
        results = data.get("results_by_k", [])

        # Extract K values and variance recovery by role data
        k_values = []
        user_vr = []
        assistant_vr = []

        for r in results:
            if "variance_recovery_by_role" not in r:
                continue

            k_values.append(r["k"])
            vr_by_role = r["variance_recovery_by_role"]

            # Focus on user vs assistant
            user_variance_recovery = vr_by_role.get("user", {}).get("variance_recovery", None)
            assistant_variance_recovery = vr_by_role.get("assistant", {}).get("variance_recovery", None)

            user_vr.append(user_variance_recovery)
            assistant_vr.append(assistant_variance_recovery)

        if not k_values:
            continue

        # Extract label from filename
        label_prefix = (
            Path(result_file).stem.replace("gemma2", "").replace("k_sweep_results_", "").replace("_", " ").strip()
        )
        if not label_prefix:
            label_prefix = f"Run {file_idx + 1}"

        line_style = line_styles[file_idx % len(line_styles)]
        marker_style = marker_styles[file_idx % len(marker_styles)]

        # Filter out None values and plot
        user_data = [(k, v) for k, v in zip(k_values, user_vr) if v is not None]
        if user_data:
            k_vals, vr_vals = zip(*user_data)
            plt.plot(
                k_vals,
                vr_vals,
                marker=marker_style,
                linestyle=line_style,
                color="#0072B2",
                linewidth=2.5,
                markersize=8,
                label=f"{label_prefix} - User",
                alpha=0.8,
            )
            plot_has_data = True

        assistant_data = [(k, v) for k, v in zip(k_values, assistant_vr) if v is not None]
        if assistant_data:
            k_vals, vr_vals = zip(*assistant_data)
            plt.plot(
                k_vals,
                vr_vals,
                marker=marker_style,
                linestyle=line_style,
                color="#D55E00",
                linewidth=2.5,
                markersize=8,
                label=f"{label_prefix} - Assistant",
                alpha=0.8,
            )
            plot_has_data = True

    if not plot_has_data:
        plt.close()
        print("No valid variance recovery by role data to plot.")
        return

    plt.xscale("log", base=2)
    plt.xlabel("Best of N", fontsize=14)
    plt.ylabel("Variance Recovery", fontsize=14)
    plt.title("Variance Recovery by Token Role - All Runs", fontsize=16)

    # Get unique k values for x-ticks
    unique_k = sorted(
        set(
            k
            for _, data in all_data
            for r in data.get("results_by_k", [])
            if "variance_recovery_by_role" in r
            for k in [r["k"]]
        )
    )
    if unique_k:
        plt.xticks(unique_k, [str(k) for k in unique_k])

    plt.grid(True, alpha=0.3)

    # Only add legend if we have data
    if plot_has_data:
        plt.legend(loc="lower right", fontsize=10, ncol=2)

    plt.tight_layout()

    output_path = output_dir / "combined_variance_recovery_by_role.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Combined variance recovery by role plot saved to {output_path}")


def plot_all_json_files():
    """Find all gemma2 JSON files and create a combined plot."""
    eval_dir = Path(__file__).parent

    # Find all gemma2 k_sweep files
    json_files = sorted(eval_dir.glob("gemma2*.json"))
    json_files += sorted(eval_dir.glob("k_sweep_results*.json"))
    if not json_files:
        print("No gemma2*.json files found in eval_results/")
        return

    print(f"Found {len(json_files)} result files:")
    for f in json_files:
        print(f"  - {f.name}")

    # Create plot
    plot_variance_recovery(json_files, output_file="gemma2_variance_recovery.png", plot_histograms=True)

    # Also create combined MSE by role plot
    output_dir = Path(__file__).parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all data for combined plot
    all_data = []
    for result_file in json_files:
        try:
            data = load_results(result_file)
            all_data.append((result_file, data))
        except Exception as e:
            print(f"Skipping {result_file} for combined plot due to error: {e}")

    if all_data:
        plot_combined_mse_by_role(all_data, output_dir)
        plot_combined_variance_recovery_by_role(all_data, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Plot variance recovery from K-sweep results")
    parser.add_argument("files", nargs="*", help="JSON result files to plot (if none, plots all gemma2 files)")
    parser.add_argument(
        "-o",
        "--output",
        default="variance_recovery_plot.png",
        help="Output filename (default: variance_recovery_plot.png)",
    )
    parser.add_argument("--hist", action="store_true", help="Generate histograms for each result file")
    print("plotting histograms")
    args = parser.parse_args()

    if args.files:
        # Plot specific files
        plot_variance_recovery(args.files, args.output, plot_histograms=args.hist)
    else:
        # Plot all gemma2 files
        plot_all_json_files()


if __name__ == "__main__":
    main()
