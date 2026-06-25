"""
visualization.py
HydroSense-Kenya — Scientific Visualization Module

All figures follow scientific best practice:
  - Labelled axes with units
  - Titles that state the scientific question or finding
  - Legends where multiple series are present
  - Tight layout and optional save-to-file
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional


# Shared style
ZONE_COLORS = {"Zone_A": "#2196F3", "Zone_B": "#4CAF50", "Zone_C": "#FF9800"}
FIG_DPI = 120


def _savefig(fig, path: Optional[str]):
    if path:
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"[visualization] Figure saved to '{path}'")

# 1. Rainfall time series

def plot_rainfall(weather_df: pd.DataFrame, save_path: Optional[str] = None):
    """Bar chart of daily rainfall with flagged outliers highlighted."""
    fig, ax = plt.subplots(figsize=(12, 4))
    dates = weather_df["date"]
    rain = weather_df["rainfall_mm"]

    ax.bar(dates, rain, color="#1565C0", alpha=0.7, label="Rainfall (mm)")

    if "rainfall_flag" in weather_df.columns:
        flagged = weather_df[weather_df["rainfall_flag"]]
        ax.bar(flagged["date"], flagged["rainfall_mm"],
               color="red", alpha=0.9, label="Flagged outlier")

    ax.set_xlabel("Date")
    ax.set_ylabel("Rainfall (mm)")
    ax.set_title("Daily Rainfall — March 2026 | HydroSense Farm")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    _savefig(fig, save_path)
    plt.show()



# 2. Soil moisture by zone

def plot_soil_moisture_zones(soil_df: pd.DataFrame, crop_params: pd.DataFrame,
                              save_path: Optional[str] = None):
    """
    Line plot of daily soil moisture for each zone.
    Horizontal dashed lines show minimum and target moisture thresholds.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    for zone_id, group in soil_df.groupby("zone_id"):
        date = group["timestamp"].dt.normalize()
        ax.plot(date, group["soil_moisture_pct"],
                color=ZONE_COLORS.get(zone_id, "grey"),
                label=zone_id, linewidth=1.8, marker="o", markersize=3)

    # Threshold lines from crop params
    for _, row in crop_params.iterrows():
        ax.axhline(row["min_moisture_pct"], color=ZONE_COLORS.get(row["zone_id"], "grey"),
                   linestyle="--", alpha=0.5, linewidth=1)
        ax.axhline(row["target_moisture_pct"], color=ZONE_COLORS.get(row["zone_id"], "grey"),
                   linestyle=":", alpha=0.5, linewidth=1)

    ax.set_xlabel("Date")
    ax.set_ylabel("Soil Moisture (%)")
    ax.set_title("Soil Moisture by Zone — Dashed = Minimum | Dotted = Target")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    _savefig(fig, save_path)
    plt.show()


# 3. Evapotranspiration

def plot_et_series(weather_df: pd.DataFrame, et_series: np.ndarray,
                   save_path: Optional[str] = None):
    """Line plot of computed daily ET alongside temperature and rainfall."""
    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(weather_df["date"], et_series, color="#E53935",
             label="ET (mm/day)", linewidth=2)
    ax1.bar(weather_df["date"], weather_df["rainfall_mm"],
            alpha=0.3, color="#1565C0", label="Rainfall (mm)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("mm / day")
    ax1.set_title("Estimated Daily Evapotranspiration vs Rainfall")

    ax2 = ax1.twinx()
    ax2.plot(weather_df["date"], weather_df["temperature_c"],
             color="#FB8C00", linestyle="--", linewidth=1.2, label="Temp (°C)")
    ax2.set_ylabel("Temperature (°C)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    _savefig(fig, save_path)
    plt.show()


# 4. Water deficit

def plot_water_deficit(dates: np.ndarray, daily_deficit: np.ndarray,
                        save_path: Optional[str] = None):
    """Area plot showing daily and cumulative water deficit."""
    cumulative = np.cumsum(daily_deficit)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax1.fill_between(dates, daily_deficit, alpha=0.6, color="#F44336", label="Daily deficit")
    ax1.set_ylabel("Water Deficit (mm/day)")
    ax1.set_title("Daily Water Deficit (ET − Rainfall when positive)")
    ax1.legend()

    ax2.plot(dates, cumulative, color="#880E4F", linewidth=2)
    ax2.fill_between(dates, cumulative, alpha=0.3, color="#880E4F")
    ax2.set_ylabel("Cumulative Deficit (mm)")
    ax2.set_xlabel("Date")
    ax2.set_title("Cumulative Water Deficit — Trapezoidal Integration")

    fig.tight_layout()
    _savefig(fig, save_path)
    plt.show()


# 5. Simulation comparison: Euler vs RK4

def plot_simulation_comparison(days: np.ndarray,
                                euler_moisture: np.ndarray,
                                rk4_moisture: np.ndarray,
                                min_moisture: float,
                                target_moisture: float,
                                field_capacity: float,
                                zone_id: str = "",
                                save_path: Optional[str] = None):
    """Side-by-side comparison of Euler and RK4 soil moisture trajectories."""
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(days, euler_moisture, label="Euler method", color="#1976D2",
            linestyle="--", linewidth=1.8)
    ax.plot(days, rk4_moisture, label="RK4 method", color="#388E3C",
            linewidth=2.0)
    ax.axhline(min_moisture, color="red", linestyle=":", linewidth=1.2,
               label=f"Min moisture ({min_moisture}%)")
    ax.axhline(target_moisture, color="orange", linestyle=":", linewidth=1.2,
               label=f"Target ({target_moisture}%)")
    ax.axhline(field_capacity, color="grey", linestyle=":", linewidth=1.2,
               label=f"Field capacity ({field_capacity}%)")

    title = f"Soil Moisture Simulation: Euler vs RK4"
    if zone_id:
        title += f" — {zone_id}"
    ax.set_title(title)
    ax.set_xlabel("Day")
    ax.set_ylabel("Soil Moisture (%)")
    ax.legend()
    fig.tight_layout()
    _savefig(fig, save_path)
    plt.show()


# 6. Monte Carlo uncertainty band

def plot_monte_carlo_band(days: np.ndarray, percentiles: dict,
                           min_moisture: float, zone_id: str = "",
                           save_path: Optional[str] = None):
    """Uncertainty band from Monte Carlo simulation."""
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.fill_between(days, percentiles["p05"], percentiles["p95"],
                    alpha=0.25, color="#7B1FA2", label="5th–95th percentile")
    ax.plot(days, percentiles["p50"], color="#7B1FA2",
            linewidth=2, label="Median (P50)")
    ax.axhline(min_moisture, color="red", linestyle="--", linewidth=1.2,
               label=f"Stress threshold ({min_moisture}%)")

    title = "Monte Carlo Soil Moisture Uncertainty (1 000 Rainfall Scenarios)"
    if zone_id:
        title += f" — {zone_id}"
    ax.set_title(title)
    ax.set_xlabel("Day")
    ax.set_ylabel("Soil Moisture (%)")
    ax.legend()
    fig.tight_layout()
    _savefig(fig, save_path)
    plt.show()

# 7. Optimised irrigation schedule

def plot_irrigation_schedule(days_irr: np.ndarray,
                              irrigation: np.ndarray,
                              moisture: np.ndarray,
                              min_moisture: float,
                              target_moisture: float,
                              zone_id: str = "",
                              save_path: Optional[str] = None):
    """Two-panel plot: daily irrigation and resulting soil moisture."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    ax1.bar(days_irr, irrigation, color="#00897B", alpha=0.8, label="Irrigation (mm)")
    ax1.set_ylabel("Irrigation (mm)")
    title = "Optimised Irrigation Schedule"
    if zone_id:
        title += f" — {zone_id}"
    ax1.set_title(title)
    ax1.legend()

    moisture_days = np.arange(len(moisture))
    ax2.plot(moisture_days, moisture, color="#1565C0", linewidth=2, label="Soil Moisture (%)")
    ax2.axhline(min_moisture, color="red", linestyle="--",
                linewidth=1.2, label=f"Min ({min_moisture}%)")
    ax2.axhline(target_moisture, color="orange", linestyle=":",
                linewidth=1.2, label=f"Target ({target_moisture}%)")
    ax2.set_ylabel("Soil Moisture (%)")
    ax2.set_xlabel("Day")
    ax2.legend()

    fig.tight_layout()
    _savefig(fig, save_path)
    plt.show()


# 8. Convergence comparison for root-finding methods

def plot_convergence(results: dict, save_path: Optional[str] = None):
    """
    Log-scale convergence plot comparing root-finding methods.

    Parameters
    ----------
    results : dict method_name -> result dict from numerical_methods
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    styles = {"bisection": ("o-", "#1976D2"),
              "newton_raphson": ("s-", "#388E3C"),
              "secant": ("^-", "#F57C00")}

    for method, res in results.items():
        errors = [h["error"] for h in res["history"]]
        iters = [h["iter"] for h in res["history"]]
        style, color = styles.get(method, (".-", "grey"))
        ax.semilogy(iters, errors, style, color=color,
                    label=f"{method} ({res['iterations']} iters)", linewidth=1.5)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Absolute Error (log scale)")
    ax.set_title("Convergence Comparison: Root-Finding Methods")
    ax.legend()
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    fig.tight_layout()
    _savefig(fig, save_path)
    plt.show()


# 9. Descriptive statistics heatmap (bonus)

def plot_zone_stats_heatmap(stats_df: pd.DataFrame, save_path: Optional[str] = None):
    """Heatmap of mean soil moisture, ET, and rainfall by zone and week."""
    fig, ax = plt.subplots(figsize=(10, 4))
    data = stats_df.values.astype(float)
    im = ax.imshow(data, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(stats_df.columns)))
    ax.set_xticklabels(stats_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(stats_df.index)))
    ax.set_yticklabels(stats_df.index)
    plt.colorbar(im, ax=ax, label="Value")
    ax.set_title("Zone × Metric Summary Heatmap")
    fig.tight_layout()
    _savefig(fig, save_path)
    plt.show()
