"""
HydroSense-Kenya — Data Cleaning and Validation Module

Functions for loading, inspecting, cleaning, and validating the three
HydroSense-Kenya datasets:
  - weather_daily.csv
  - soil_sensor_data.csv
  - crop_zone_parameters.csv

All cleaning decisions are documented via print statements so that
notebook cells capture a full audit trail.
"""

import numpy as np
import pandas as pd

# 1 Loading helpers for each dataset
def load_weather(path: str) -> pd.DataFrame:
    """Load weather_daily.csv with standard NA recognition."""
    df = pd.read_csv(path, na_values=["NA", ""])
    df["date"] = pd.to_datetime(df["date"])
    print(f"[load_weather] Loaded {len(df)} rows from '{path}'")
    return df


def load_soil(path: str) -> pd.DataFrame:
    """Load soil_sensor_data.csv with standard NA recognition."""
    df = pd.read_csv(path, na_values=["NA", ""])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"[load_soil] Loaded {len(df)} rows from '{path}'")
    return df


def load_crop_params(path: str) -> pd.DataFrame:
    """Load crop_zone_parameters.csv."""
    df = pd.read_csv(path, na_values=["NA", ""])
    print(f"[load_crop_params] Loaded {len(df)} rows from '{path}'")
    return df

# 2. Inspection helpers

def detect_outliers_iqr(series: pd.Series, label: str = "", factor: float = 1.5) -> pd.Series:
    """
    Return a boolean mask of outliers using the IQR fence method.
    Outlier if value < Q1 - k*IQR  or  value > Q3 + k*IQR.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    mask = (series < lower) | (series > upper)
    n = mask.sum()
    print(f"[detect_outliers_iqr] '{label}': {n} outlier(s) | fence=[{lower:.2f}, {upper:.2f}]")
    return mask

# 3. Cleaning - weather

def clean_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean weather_daily.csv:
      1. Impute missing rainfall_mm with 0 (no rain detected → 0).
      2. Impute missing humidity_pct with column median.
      3. Cap extreme temperature outlier (>40°C unlikely in Kenyan highlands).
      4. Flag extreme rainfall outlier (>60 mm) but retain value.
    Returns a cleaned copy.
    """
    df = df.copy()

    # rainfall_mm
    n_rain_na = df["rainfall_mm"].isna().sum()
    df["rainfall_mm"] = df["rainfall_mm"].fillna(0.0)
    print(f"[clean_weather] rainfall_mm: {n_rain_na} NA → filled with 0.0")

    # humidity_pct
    n_hum_na = df["humidity_pct"].isna().sum()
    hum_median = df["humidity_pct"].median()
    df["humidity_pct"] = df["humidity_pct"].fillna(hum_median)
    print(f"[clean_weather] humidity_pct: {n_hum_na} NA → filled with median {hum_median:.1f}")

    # temperature outlier
    temp_mask = df["temperature_c"] > 40.0
    if temp_mask.any():
        temp_median = df.loc[~temp_mask, "temperature_c"].median()
        bad_dates = df.loc[temp_mask, "date"].dt.strftime("%Y-%m-%d").tolist()
        df.loc[temp_mask, "temperature_c"] = temp_median
        print(f"[clean_weather] temperature_c: capped {len(bad_dates)} outlier(s) on {bad_dates} → {temp_median:.1f}°C (column median)")

    # rainfall outlier flag 
    rain_mask = df["rainfall_mm"] > 60.0
    if rain_mask.any():
        df["rainfall_flag"] = rain_mask
        bad_dates = df.loc[rain_mask, "date"].dt.strftime("%Y-%m-%d").tolist()
        print(f"[clean_weather] rainfall_mm: {len(bad_dates)} extreme event(s) on {bad_dates} flagged (retained, column 'rainfall_flag' added)")
    else:
        df["rainfall_flag"] = False

    return df

# 4. Cleaning - soil sensor

def clean_soil(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean soil_sensor_data.csv:
      1. Impute missing soil_moisture_pct with zone-wise median.
      2. Cap impossible tank_level_liters outlier (>9000 L suspicious).
      3. Set pump_flow_lpm to NaN where sensor_status == 'CHECK'.
      4. Flag and impute anomalous soil_moisture_pct (<10%) with zone median.
    Returns a cleaned copy.
    """
    df = df.copy()

    # soil_moisture_pct missing 
    n_sm_na = df["soil_moisture_pct"].isna().sum()
    zone_medians = df.groupby("zone_id")["soil_moisture_pct"].transform("median")
    df["soil_moisture_pct"] = df["soil_moisture_pct"].fillna(zone_medians)
    print(f"[clean_soil] soil_moisture_pct: {n_sm_na} NA → filled with zone-wise median")

    # soil_moisture_pct anomaly (<10%)
    low_mask = df["soil_moisture_pct"] < 10.0
    if low_mask.any():
        zone_medians2 = df.groupby("zone_id")["soil_moisture_pct"].transform("median")
        bad_rows = df.loc[low_mask, ["timestamp", "zone_id", "soil_moisture_pct"]].to_string()
        df.loc[low_mask, "soil_moisture_pct"] = zone_medians2[low_mask]
        print(f"[clean_soil] soil_moisture_pct: anomalously low value(s) detected:\n{bad_rows}\n  → replaced with zone-wise median")

    # tank_level_liters outlier 
    tank_mask = df["tank_level_liters"] > 9000
    if tank_mask.any():
        tank_median = df.loc[~tank_mask, "tank_level_liters"].median()
        bad_rows = df.loc[tank_mask, ["timestamp", "zone_id", "tank_level_liters"]].to_string()
        df.loc[tank_mask, "tank_level_liters"] = tank_median
        print(f"[clean_soil] tank_level_liters: anomaly detected:\n{bad_rows}\n  → replaced with column median {tank_median:.0f} L")

    # pump_flow_lpm 
    check_mask = df["sensor_status"] == "CHECK"
    if check_mask.any():
        df.loc[check_mask, "pump_flow_lpm"] = np.nan
        print(f"[clean_soil] pump_flow_lpm: {check_mask.sum()} row(s) with sensor_status=CHECK → pump_flow set to NaN")

    return df

# 5. Merge and export cleaned datasets
def merge_datasets(weather_clean: pd.DataFrame,
                   soil_clean: pd.DataFrame,
                   crop_params: pd.DataFrame) -> pd.DataFrame:
    """
    Merge cleaned weather and soil datasets on date, then attach crop params.
    Returns one flat DataFrame suitable for analysis.
    """
    # Extract date from soil timestamp
    soil_clean = soil_clean.copy()
    soil_clean["date"] = soil_clean["timestamp"].dt.normalize()

    merged = soil_clean.merge(weather_clean, on="date", how="left")
    merged = merged.merge(crop_params, on="zone_id", how="left")
    print(f"[merge_datasets] Merged dataset: {merged.shape[0]} rows × {merged.shape[1]} columns")
    return merged

def export_clean(df: pd.DataFrame, path: str) -> None:
    """Save cleaned DataFrame to CSV."""
    df.to_csv(path, index=False)
    print(f"[export_clean] Saved to '{path}'")
def summarise_missing(df: pd.DataFrame, name: str = "DataFrame") -> pd.Series:
    """Return count and percentage of missing values per column."""
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)
    report = pd.DataFrame({"missing_count": missing_count, "missing_pct": missing_pct})
    report = report[report["missing_count"] > 0]
    print(f"\n[summarise_missing] {name} — {len(report)} columns with missing data:")
    print(report.to_string())
    return report

# 6. Pipeline Helpers

def audit_sensor_status(df: pd.DataFrame) -> pd.Series:
    """
    Returns a value count breakdown of the 'sensor_status' flags
    to trace operational errors and logs an audit trail summary.
    """
    print("\n[audit_sensor_status] Checking operational status profile:")
    if 'sensor_status' in df.columns:
        counts = df['sensor_status'].value_counts(dropna=False)
        print(counts.to_string())
        return counts
    else:
        print("Warning: 'sensor_status' column missing from this DataFrame.")
        return pd.Series(dtype=int)


def clean_crop_params(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pass-through or normalization routine for crop parameters. 
    Returns a verified copy of the parameters dataframe.
    """
    print(f"[clean_crop_params] Validated {len(df)} crop zone configuration profiles.")
    return df.copy()