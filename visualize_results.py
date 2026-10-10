"""
visualize_results.py - High-precision publication-grade visualization suite for BJORK.
Strict Color Palette:
- Primary: #2373F4 and its variants (#0B2C68, #1451B8, #5B97F7, #9DC0FA, #E6EFFF)
- Grayscale: Neutral dark slate (#0B0F17, #1A202C, #2D3748, #718096, #E2E8F0, #FFFFFF)

Generates:
1. Trajectory Learned vs Initial comparison with gradient & slew waveforms.
2. Ground Truth vs Zero-Filled vs CG-SENSE vs BJORK Reconstructions with amplified Error Maps.
3. Region of Interest (ROI) Zoom Detailed Map with 1D intensity edge profile.
4. Training Convergence & Trajectory Optimization Dynamics.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless rendering suitable for cluster and local runs
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches


# ==============================================================================
# Strict Theme & Color Palette (#2373F4, variants, and gray)
# ==============================================================================
COLOR_PRIMARY = "#2373F4"       # Primary electric royal blue
COLOR_DARK_BLUE = "#1451B8"     # Darker blue variant
COLOR_DEEP_NAVY = "#0B2C68"     # Deep navy blue variant
COLOR_LIGHT_BLUE = "#5B97F7"    # Lighter blue variant
COLOR_ICE_BLUE = "#9DC0FA"      # Soft ice blue variant
COLOR_TINT_BLUE = "#E6EFFF"     # Pale blue wash

# Grayscale palette
COLOR_BG_DARK = "#0B0F17"       # Canvas background
COLOR_CARD_DARK = "#151B26"     # Axes card surface
COLOR_BORDER = "#283347"        # Subplot border & subtle lines
COLOR_TEXT_MUTED = "#8896AB"    # Minor annotations
COLOR_TEXT_LIGHT = "#E2E8F0"    # Major labels
COLOR_GRID = "#1E2738"          # Grid lines

# Custom Error Colormap: dark slate -> dark blue -> #2373F4 -> bright ice blue
CMAP_ERROR = LinearSegmentedColormap.from_list(
    "bjork_error",
    [
        (0.00, "#0B0F17"),
        (0.20, "#0F2B5C"),
        (0.50, "#1A52B8"),
        (0.80, "#2373F4"),
        (1.00, "#D9E7FD"),
    ],
    N=256,
)

# Custom Trajectory Colormap: deep navy -> #2373F4 -> bright highlight
CMAP_TRAJ = LinearSegmentedColormap.from_list(
    "bjork_traj",
    [
        (0.00, "#0B2C68"),
        (0.35, "#1451B8"),
        (0.70, "#2373F4"),
        (1.00, "#A8CCFA"),
    ],
    N=256,
)


def apply_theme(fig, axes):
    """Apply consistent high-contrast dark theme using grayscale and #2373F4."""
    fig.patch.set_facecolor(COLOR_BG_DARK)
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    for ax in np.array(axes).flatten():
        ax.set_facecolor(COLOR_CARD_DARK)
        for spine in ax.spines.values():
            spine.set_color(COLOR_BORDER)
            spine.set_linewidth(1.0)
        ax.tick_params(colors=COLOR_TEXT_MUTED, labelsize=9)
        ax.xaxis.label.set_color(COLOR_TEXT_LIGHT)
        ax.yaxis.label.set_color(COLOR_TEXT_LIGHT)
        ax.title.set_color(COLOR_TEXT_LIGHT)


# ==============================================================================
# Plot 1: Trajectory Comparison & Hardware Waveform Map
# ==============================================================================
def plot_trajectory_detailed_map(
    traj_initial,
    traj_learned,
    grad_learned=None,
    slew_learned=None,
    g_max=4.0,
    s_max=15000.0,
    dt=4e-6,
    save_path="trajectory_analysis.png",
):
    """
    Plots:
    - Initial baseline trajectory (e.g., radial spokes) in k-space.
    - Learned BJORK trajectory in k-space.
    - Gradient amplitude waveform |G(t)| with G_max limit.
    - Slew rate waveform |S(t)| with S_max limit.
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
    apply_theme(fig, axs)

    # 1. Initial Trajectory
    ax_init = axs[0, 0]
    # traj shape: (2, num_points) or (2, num_shots, nfe)
    ti = np.reshape(traj_initial, (2, -1))
    ax_init.scatter(ti[0], ti[1], s=0.2, c=COLOR_TEXT_MUTED, alpha=0.6, rasterized=True)
    ax_init.set_title("Standard Baseline Trajectory (k-space)", fontsize=12, fontweight="bold", pad=10)
    ax_init.set_xlim([-np.pi * 1.05, np.pi * 1.05])
    ax_init.set_ylim([-np.pi * 1.05, np.pi * 1.05])
    ax_init.set_xlabel("$k_x$ (rad)")
    ax_init.set_ylabel("$k_y$ (rad)")
    ax_init.set_aspect("equal")
    ax_init.grid(True, color=COLOR_GRID, linestyle="--", linewidth=0.6)

    # 2. Learned BJORK Trajectory
    ax_opt = axs[0, 1]
    to = np.reshape(traj_learned, (2, -1))
    # Color by radius to show coverage distribution
    radii = np.sqrt(to[0] ** 2 + to[1] ** 2)
    sc = ax_opt.scatter(to[0], to[1], s=0.3, c=radii, cmap=CMAP_TRAJ, alpha=0.85, rasterized=True)
    ax_opt.set_title("Learned BJORK Trajectory (k-space)", fontsize=12, fontweight="bold", pad=10)
    ax_opt.set_xlim([-np.pi * 1.05, np.pi * 1.05])
    ax_opt.set_ylim([-np.pi * 1.05, np.pi * 1.05])
    ax_opt.set_xlabel("$k_x$ (rad)")
    ax_opt.set_ylabel("$k_y$ (rad)")
    ax_opt.set_aspect("equal")
    ax_opt.grid(True, color=COLOR_GRID, linestyle="--", linewidth=0.6)

    # Unit boundary circle
    circle = patches.Circle((0, 0), np.pi, fill=False, edgecolor=COLOR_PRIMARY, linewidth=1.2, linestyle=":")
    ax_opt.add_patch(circle)

    # 3. Gradient Waveforms G(t)
    ax_grad = axs[1, 0]
    if grad_learned is not None:
        g = np.abs(grad_learned).reshape(-1)
        time_ms = np.arange(len(g)) * dt * 1e3
        ax_grad.plot(time_ms, g, color=COLOR_PRIMARY, linewidth=1.0, label="|G(t)| Learned")
        ax_grad.axhline(g_max, color=COLOR_TEXT_LIGHT, linestyle="--", linewidth=1.2, label=f"$G_{{max}} = {g_max}$ G/cm")
        ax_grad.fill_between(time_ms, g_max, np.maximum(g, g_max), color=COLOR_LIGHT_BLUE, alpha=0.3, label="Violation Penalty")
        ax_grad.set_ylabel("Gradient (G/cm)")
        ax_grad.set_xlabel("Time (ms)")
        ax_grad.set_title("Gradient Amplitude Waveform |G(t)|", fontsize=12, fontweight="bold", pad=10)
        ax_grad.legend(facecolor=COLOR_CARD_DARK, edgecolor=COLOR_BORDER, labelcolor=COLOR_TEXT_LIGHT, fontsize=9)
        ax_grad.grid(True, color=COLOR_GRID, linestyle="--", linewidth=0.6)
    else:
        ax_grad.text(0.5, 0.5, "Gradient data not supplied", ha="center", va="center", color=COLOR_TEXT_MUTED)

    # 4. Slew Rate Waveforms S(t)
    ax_slew = axs[1, 1]
    if slew_learned is not None:
        s = np.abs(slew_learned).reshape(-1)
        time_ms = np.arange(len(s)) * dt * 1e3
        ax_slew.plot(time_ms, s, color=COLOR_LIGHT_BLUE, linewidth=1.0, label="|S(t)| Learned")
        ax_slew.axhline(s_max, color=COLOR_TEXT_LIGHT, linestyle="--", linewidth=1.2, label=f"$S_{{max}} = {s_max}$ G/cm/s")
        ax_slew.fill_between(time_ms, s_max, np.maximum(s, s_max), color=COLOR_PRIMARY, alpha=0.3, label="Violation Penalty")
        ax_slew.set_ylabel("Slew Rate (G/cm/s)")
        ax_slew.set_xlabel("Time (ms)")
        ax_slew.set_title("Slew Rate Waveform |S(t)|", fontsize=12, fontweight="bold", pad=10)
        ax_slew.legend(facecolor=COLOR_CARD_DARK, edgecolor=COLOR_BORDER, labelcolor=COLOR_TEXT_LIGHT, fontsize=9)
        ax_slew.grid(True, color=COLOR_GRID, linestyle="--", linewidth=0.6)
    else:
        ax_slew.text(0.5, 0.5, "Slew rate data not supplied", ha="center", va="center", color=COLOR_TEXT_MUTED)

    plt.suptitle("BJORK K-Space Trajectory & Hardware Waveform Analysis", fontsize=15, fontweight="bold", color=COLOR_TEXT_LIGHT, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved trajectory analysis plot to: {save_path}")


# ==============================================================================
# Plot 2: Comprehensive Reconstruction & Amplified Error Map
# ==============================================================================
def plot_reconstruction_error_map(
    target_img,
    recon_dict,
    metrics_dict=None,
    error_scale=5.0,
    save_path="reconstruction_comparison_error_map.png",
):
    """
    Detailed grid:
    Top row: Magnitude images (Ground Truth, Zero-Filled, CG-SENSE, BJORK MoDL).
    Bottom row: Amplified absolute error maps |Recon - Target| * error_scale with #2373F4 colormap.
    """
    methods = list(recon_dict.keys())
    ncols = 1 + len(methods)
    fig, axs = plt.subplots(2, ncols, figsize=(4.2 * ncols, 8.5), dpi=300)
    apply_theme(fig, axs)

    targ_mag = np.abs(target_img).squeeze()
    max_val = max(targ_mag.max(), 1e-8)
    targ_norm = targ_mag / max_val

    # Top-Left: Reference Image
    axs[0, 0].imshow(targ_norm, cmap="gray", vmin=0, vmax=1)
    axs[0, 0].set_title("Ground Truth Reference", fontsize=11, fontweight="bold", pad=8)
    axs[0, 0].axis("off")

    # Bottom-Left: Reference Blank Card with Colorbar description
    axs[1, 0].text(
        0.5, 0.5,
        f"Amplified Error Maps\n({error_scale:.0f}x Absolute Error)\n\nMin: 0.0\nMax: {1.0/error_scale:.2f}",
        ha="center", va="center",
        color=COLOR_PRIMARY,
        fontsize=11,
        fontweight="bold",
        linespacing=1.5,
    )
    axs[1, 0].axis("off")

    # Plot each method
    for col_idx, method_name in enumerate(methods, start=1):
        recon_mag = np.abs(recon_dict[method_name]).squeeze() / max_val
        err = np.abs(recon_mag - targ_norm) * error_scale

        # Metric string overlay
        metric_str = ""
        if metrics_dict and method_name in metrics_dict:
            m = metrics_dict[method_name]
            metric_str = f"\nPSNR: {m.get('psnr', 0.0):.2f} dB | SSIM: {m.get('ssim', 0.0):.3f}"

        # Top: Reconstructed image
        ax_img = axs[0, col_idx]
        ax_img.imshow(recon_mag, cmap="gray", vmin=0, vmax=1)
        ax_img.set_title(f"{method_name}{metric_str}", fontsize=10, fontweight="bold", pad=8)
        ax_img.axis("off")

        # Bottom: Error map
        ax_err = axs[1, col_idx]
        im_err = ax_err.imshow(err, cmap=CMAP_ERROR, vmin=0, vmax=1)
        ax_err.set_title(f"Error ({method_name})", fontsize=10, fontweight="bold", pad=8)
        ax_err.axis("off")

    # Shared colorbar for error maps
    cbar_ax = fig.add_axes([0.92, 0.12, 0.015, 0.32])
    cbar = fig.colorbar(im_err, cax=cbar_ax)
    cbar.set_label(f"Absolute Error ({error_scale:.0f}x)", color=COLOR_TEXT_LIGHT, fontsize=10)
    cbar.ax.yaxis.set_tick_params(color=COLOR_BORDER)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=COLOR_TEXT_MUTED, fontsize=8)

    plt.suptitle("Knee MRI Reconstruction & Quantitative Error Map Benchmark", fontsize=15, fontweight="bold", color=COLOR_TEXT_LIGHT, y=0.98)
    plt.tight_layout(rect=[0, 0, 0.91, 0.95])
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved reconstruction error map to: {save_path}")


# ==============================================================================
# Plot 3: Region of Interest (ROI) Zoom Map & 1D Line Profile
# ==============================================================================
def plot_roi_detailed_map(
    target_img,
    recon_dict,
    roi_box=None,
    line_y=None,
    save_path="roi_zoom_and_line_profile.png",
):
    """
    In-depth anatomical inspection:
    - Overview image with highlighted ROI bounding box in #2373F4.
    - Magnified insets of the ROI across all methods.
    - 1D intensity cross-section profile comparing edge preservation.
    """
    targ_mag = np.abs(target_img).squeeze()
    max_val = max(targ_mag.max(), 1e-8)
    targ_norm = targ_mag / max_val
    h, w = targ_norm.shape

    # Default ROI: central knee cartilage/meniscus region
    if roi_box is None:
        y_start, y_end = int(0.40 * h), int(0.65 * h)
        x_start, x_end = int(0.35 * w), int(0.65 * w)
    else:
        y_start, y_end, x_start, x_end = roi_box

    if line_y is None:
        line_y = (y_start + y_end) // 2

    methods = list(recon_dict.keys())
    num_methods = len(methods)
    fig = plt.figure(figsize=(15, 10), dpi=300)
    apply_theme(fig, fig.axes)

    # 1. Overview image with ROI bounding box
    ax_overview = fig.add_subplot(2, num_methods + 1, 1)
    apply_theme(fig, ax_overview)
    ax_overview.imshow(targ_norm, cmap="gray", vmin=0, vmax=1)
    rect = patches.Rectangle(
        (x_start, y_start),
        x_end - x_start,
        y_end - y_start,
        linewidth=2.0,
        edgecolor=COLOR_PRIMARY,
        facecolor="none",
        linestyle="-",
    )
    ax_overview.add_patch(rect)
    ax_overview.axhline(line_y, color=COLOR_LIGHT_BLUE, linestyle="--", linewidth=1.2, alpha=0.8)
    ax_overview.set_title("Overview (ROI Box)", fontsize=10, fontweight="bold", pad=8)
    ax_overview.axis("off")

    # 2. Reference ROI Zoom
    ax_ref_zoom = fig.add_subplot(2, num_methods + 1, 2)
    apply_theme(fig, ax_ref_zoom)
    roi_targ = targ_norm[y_start:y_end, x_start:x_end]
    ax_ref_zoom.imshow(roi_targ, cmap="gray", vmin=0, vmax=1)
    ax_ref_zoom.set_title("Target (Zoomed)", fontsize=10, fontweight="bold", pad=8)
    ax_ref_zoom.axis("off")
    for spine in ax_ref_zoom.spines.values():
        spine.set_color(COLOR_PRIMARY)
        spine.set_linewidth(1.5)

    # 3. Method Zoom Insets
    for idx, name in enumerate(methods, start=3):
        ax_zoom = fig.add_subplot(2, num_methods + 1, idx)
        apply_theme(fig, ax_zoom)
        r_mag = np.abs(recon_dict[name]).squeeze() / max_val
        roi_recon = r_mag[y_start:y_end, x_start:x_end]
        ax_zoom.imshow(roi_recon, cmap="gray", vmin=0, vmax=1)
        ax_zoom.set_title(f"{name} (Zoom)", fontsize=10, fontweight="bold", pad=8)
        ax_zoom.axis("off")

    # 4. 1D Line Profile across the row line_y
    ax_line = fig.add_subplot(2, 1, 2)
    apply_theme(fig, ax_line)
    x_coords = np.arange(w)

    # Plot Reference line in white/light gray
    ax_line.plot(x_coords, targ_norm[line_y, :], color=COLOR_TEXT_LIGHT, linewidth=2.0, label="Ground Truth Reference")

    # Palette of blues and grays for methods
    method_colors = [COLOR_BORDER, COLOR_DARK_BLUE, COLOR_PRIMARY]
    for idx, name in enumerate(methods):
        col = method_colors[idx % len(method_colors)]
        r_norm = (np.abs(recon_dict[name]).squeeze() / max_val)[line_y, :]
        lw = 2.2 if "BJORK" in name else 1.2
        ls = "-" if "BJORK" in name else "--"
        ax_line.plot(x_coords, r_norm, color=col, linewidth=lw, linestyle=ls, label=name)

    # Highlight the ROI segment along the x-axis
    ax_line.axvspan(x_start, x_end, color=COLOR_DEEP_NAVY, alpha=0.35, label="Zoomed ROI Span")
    ax_line.set_title(f"1D Intensity Edge Profile (Cross-Section along Row y = {line_y})", fontsize=12, fontweight="bold", pad=10)
    ax_line.set_xlabel("Pixel Column Coordinate (x)")
    ax_line.set_ylabel("Normalized Intensity")
    ax_line.set_ylim([-0.05, 1.10])
    ax_line.grid(True, color=COLOR_GRID, linestyle="--", linewidth=0.6)
    ax_line.legend(facecolor=COLOR_CARD_DARK, edgecolor=COLOR_BORDER, labelcolor=COLOR_TEXT_LIGHT, fontsize=9, loc="upper right")

    plt.suptitle("Fine Anatomical Structure & Edge Preservation Detail Map", fontsize=15, fontweight="bold", color=COLOR_TEXT_LIGHT, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved ROI detailed map to: {save_path}")


# ==============================================================================
# Plot 4: Training Convergence Dynamics Map
# ==============================================================================
def plot_training_dynamics(
    log_csv_path,
    save_path="training_convergence_map.png",
):
    """
    Plots training loss and validation progression curves from CSV log.
    Curves formatted with #2373F4 and clean grayscale.
    """
    if not os.path.exists(log_csv_path):
        print(f"Log file not found: {log_csv_path}")
        return

    data = np.genfromtxt(log_csv_path, delimiter=",", names=True)
    epochs = data["epoch"]

    fig, axs = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    apply_theme(fig, axs)

    # 1. Total & Reconstruction Loss
    axs[0].plot(epochs, data["total_loss"], color=COLOR_PRIMARY, linewidth=2.0, label="Total Loss")
    axs[0].plot(epochs, data["recon_loss"], color=COLOR_TEXT_LIGHT, linewidth=1.5, linestyle="--", label="Recon Loss ($\mathcal{L}_{recon}$)")
    axs[0].set_title("Loss Convergence", fontsize=12, fontweight="bold", pad=10)
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].grid(True, color=COLOR_GRID, linestyle="--", linewidth=0.6)
    axs[0].legend(facecolor=COLOR_CARD_DARK, edgecolor=COLOR_BORDER, labelcolor=COLOR_TEXT_LIGHT, fontsize=9)

    # 2. Hardware Penalty Violations
    axs[1].plot(epochs, data["grad_loss"], color=COLOR_LIGHT_BLUE, linewidth=1.8, label="Grad Violation ($\mathcal{L}_G$)")
    axs[1].plot(epochs, data["slew_loss"], color=COLOR_ICE_BLUE, linewidth=1.8, linestyle=":", label="Slew Violation ($\mathcal{L}_S$)")
    axs[1].set_title("Hardware Constraints Dynamics", fontsize=12, fontweight="bold", pad=10)
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Penalty Value")
    axs[1].set_yscale("log")
    axs[1].grid(True, color=COLOR_GRID, linestyle="--", linewidth=0.6)
    axs[1].legend(facecolor=COLOR_CARD_DARK, edgecolor=COLOR_BORDER, labelcolor=COLOR_TEXT_LIGHT, fontsize=9)

    # 3. Validation PSNR / NMSE
    ax3 = axs[2]
    ax3_twin = ax3.twinx()
    ax3_twin.tick_params(colors=COLOR_TEXT_MUTED, labelsize=9)
    for spine in ax3_twin.spines.values():
        spine.set_color(COLOR_BORDER)

    l1 = ax3.plot(epochs, data["val_psnr"], color=COLOR_PRIMARY, linewidth=2.0, label="Val PSNR (dB)")
    l2 = ax3_twin.plot(epochs, data["val_nmse"], color=COLOR_TEXT_MUTED, linewidth=1.5, linestyle="--", label="Val NMSE")
    ax3.set_title("Reconstruction Quality Progression", fontsize=12, fontweight="bold", pad=10)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("PSNR (dB)", color=COLOR_PRIMARY)
    ax3_twin.set_ylabel("NMSE", color=COLOR_TEXT_MUTED)
    ax3.grid(True, color=COLOR_GRID, linestyle="--", linewidth=0.6)

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, facecolor=COLOR_CARD_DARK, edgecolor=COLOR_BORDER, labelcolor=COLOR_TEXT_LIGHT, fontsize=9)

    plt.suptitle("BJORK Training Progression & Convergence Analysis", fontsize=15, fontweight="bold", color=COLOR_TEXT_LIGHT, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved training dynamics map to: {save_path}")


def visualize_experiment_run(results_dir):
    """
    Loads genuine saved .npy reconstruction arrays and metrics.json from real experiment run
    and renders publication-grade figures using the #2373F4 and gray palette.
    """
    import json
    recon_dir = os.path.join(results_dir, "reconstructions")
    metrics_json_path = os.path.join(results_dir, "metrics.json")

    if not os.path.exists(recon_dir):
        print(f"Reconstructions directory not found: {recon_dir}")
        return

    target_files = [f for f in os.listdir(recon_dir) if f.startswith("target_") and f.endswith(".npy")]
    if not target_files:
        print(f"No target reconstruction files found in {recon_dir}")
        return

    target_file = target_files[0]
    stem = target_file.replace("target_", "").replace(".npy", "")
    targ_path = os.path.join(recon_dir, f"target_{stem}.npy")
    zf_path = os.path.join(recon_dir, f"zerofill_{stem}.npy")
    recon_path = os.path.join(recon_dir, f"trained_recon_{stem}.npy")

    print(f"Loading real knee reconstruction arrays for slice: {stem}")
    targ = np.load(targ_path)
    zf = np.load(zf_path)
    recon = np.load(recon_path)

    # Detect number of epochs from configuration
    epochs_label = "30 Ep"
    cfg_path = os.path.join(results_dir, "run_config.yaml")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r") as f:
                import yaml
                run_cfg = yaml.safe_load(f)
                num_ep = run_cfg.get("training", {}).get("epochs", 30)
                epochs_label = f"{num_ep} Ep"
        except Exception:
            pass

    recon_label = f"Trained MoDL ({epochs_label})"

    metrics = {}
    if os.path.exists(metrics_json_path):
        with open(metrics_json_path, "r") as f:
            data = json.load(f)
            metrics = {
                "Zero-Filled (R=4)": {
                    "psnr": data.get("zero_filled_baseline", {}).get("mean_psnr_db", 0.0),
                    "ssim": data.get("zero_filled_baseline", {}).get("mean_ssim", 0.0),
                    "nmse": data.get("zero_filled_baseline", {}).get("mean_nmse", 0.0),
                },
                recon_label: {
                    "psnr": data.get("trained_reconstruction", {}).get("mean_psnr_db", 0.0),
                    "ssim": data.get("trained_reconstruction", {}).get("mean_ssim", 0.0),
                    "nmse": data.get("trained_reconstruction", {}).get("mean_nmse", 0.0),
                }
            }

    recons_dict = {
        "Zero-Filled (R=4)": zf,
        recon_label: recon,
    }

    out_recon_map = os.path.join(results_dir, "reconstruction_comparison_error_map.png")
    plot_reconstruction_error_map(targ, recons_dict, metrics, error_scale=5.0, save_path=out_recon_map)

    out_roi_map = os.path.join(results_dir, "roi_zoom_and_line_profile.png")
    plot_roi_detailed_map(targ, recons_dict, save_path=out_roi_map)

    print("\n" + "=" * 65)
    print(" Visualization Suite Generated Successfully! ")
    print("=" * 65)
    print(f"  1. Reconstruction & Error Map : {out_recon_map}")
    print(f"  2. ROI Zoom & Profile Map     : {out_roi_map}")
    print("=" * 65)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Render publication visualizations")
    parser.add_argument("--results-dir", type=str, default="results/knee_first_run",
                        help="Path to results directory containing metrics.json and reconstructions/")
    args = parser.parse_args()

    if os.path.exists(os.path.join(args.results_dir, "reconstructions")):
        visualize_experiment_run(args.results_dir)
    else:
        print(f"No experiment reconstructions found in {args.results_dir}. Running self-test...")
        from traj_utils import generate_radial_trajectory, generate_spiral_trajectory

        ti = generate_radial_trajectory(num_shots=16, nfe=256, golden_angle=True)
        to = generate_spiral_trajectory(num_shots=16, nfe=256)
        grad = np.random.uniform(0.5, 3.8, (16, 64))
        slew = np.random.uniform(2000, 14000, (16, 64))
        plot_trajectory_detailed_map(ti, to, grad, slew, save_path="test_plots/trajectory_analysis.png")

        targ_mag = np.random.rand(128, 128).astype(np.float32)
        recon_dict = {
            "Zero-Filled": np.clip(targ_mag + 0.12 * np.random.randn(*targ_mag.shape), 0, None),
            "Radial CG-SENSE": np.clip(targ_mag + 0.05 * np.random.randn(*targ_mag.shape), 0, None),
            "BJORK MoDL": np.clip(targ_mag + 0.015 * np.random.randn(*targ_mag.shape), 0, None),
        }
        metrics_dict = {
            "Zero-Filled": {"psnr": 22.45, "ssim": 0.612, "nmse": 0.128},
            "Radial CG-SENSE": {"psnr": 29.80, "ssim": 0.835, "nmse": 0.024},
            "BJORK MoDL": {"psnr": 34.12, "ssim": 0.928, "nmse": 0.008},
        }
        plot_reconstruction_error_map(targ_mag, recon_dict, metrics_dict, error_scale=5.0, save_path="test_plots/reconstruction_comparison_error_map.png")
        plot_roi_detailed_map(targ_mag, recon_dict, save_path="test_plots/roi_zoom_and_line_profile.png")
        print("Test plots generated successfully in ./test_plots/!")

