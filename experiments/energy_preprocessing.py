"""
energy_preprocessing.py
ENTSO-E Data Preprocessing Pipeline
Produces two datasets that mirror your existing weather/extreme weather tasks:
  1. energy_forecast.json   — short-term load forecasting (mirrors weather task)
  2. energy_extreme.json    — extreme demand event detection (mirrors extreme weather task)

Setup:
    pip install entsoe-py pandas numpy
    Register at https://transparency.entsoe.eu → request API key via email
"""

import json
import sys
import numpy as np
import pandas as pd
from datetime import timedelta

# --- Defensive import with clear diagnosis ---
try:
    from entsoe import EntsoePandasClient
except ImportError as e:
    print("\n❌ ImportError: Cannot import EntsoePandasClient.")
    print("   Likely cause: outdated or broken entsoe-py install in your conda env.")
    print("\n   Fix (run in terminal, with your env active):")
    print("     pip uninstall entsoe-py -y")
    print("     pip install entsoe-py --upgrade --no-cache-dir")
    print("\n   If that fails (conda conflict), try:")
    print("     conda install -c conda-forge entsoe-py")
    print("\n   Then verify with:")
    print("     python -c \"from entsoe import EntsoePandasClient; print('OK')\"")
    print(f"\n   Original error: {e}")
    sys.exit(1)

# ─────────────────────────────────────────
# CONFIG — edit these
# ─────────────────────────────────────────
API_KEY     = "69f93dc4-962c-4849-bc09-63e5a27bbc15"   # from transparency.entsoe.eu account
COUNTRY     = "DE_LU"                 # Germany-Luxembourg (stable, rich data)
START       = pd.Timestamp("20230101", tz="UTC")
END         = pd.Timestamp("20240101", tz="UTC")
HISTORY_LEN = 48   # 48 x 30min = 24h of history fed to LLM as input
PRED_LEN    = 48   # 48 x 30min = 24h to predict
EXTREME_STD  = 1.5  # mean + 1.5σ ≈ top 7% — more positives while still meaningful
TARGET_TOTAL = 100  # desired final dataset size
BALANCE      = True # if True, undersample negatives to match positives


# ─────────────────────────────────────────
# STEP 1: Download from ENTSO-E
# ─────────────────────────────────────────
def download_load(api_key, country, start, end):
    """
    Downloads actual total load (MW) at 30-min resolution.
    Returns a cleaned pandas Series indexed by UTC timestamp.

    ENTSO-E fields returned:
      - index: DatetimeTZDtype (UTC timestamps, 30-min intervals)
      - values: float (Actual Load in MW)
    """
    client = EntsoePandasClient(api_key=api_key)
    print(f"Downloading actual load for {country} from {start.date()} to {end.date()}...")

    # query_load returns actual total load [6.1.A]
    # This is the primary numerical target — equivalent to "temperature" in weather
    load = client.query_load(country, start=start, end=end)

    # entsoe-py returns a DataFrame with column 'Actual Load'
    if isinstance(load, pd.DataFrame):
        load = load["Actual Load"]

    load.name = "load_mw"
    print(f"  Downloaded {len(load)} records, resolution: {_infer_resolution(load)}")
    return load


def _infer_resolution(series):
    diffs = series.index.to_series().diff().dropna().unique()
    return str(diffs[0]) if len(diffs) > 0 else "unknown"


# ─────────────────────────────────────────
# STEP 2: Clean the raw load series
# ─────────────────────────────────────────
def clean_load(load: pd.Series) -> pd.Series:
    """
    Handles the three most common ENTSO-E data quality issues:
      1. Missing timestamps (reindex to fill gaps)
      2. Missing values (linear interpolation, max 4 gaps = 2h)
      3. Outliers from measurement errors (3-sigma rule with rolling window)
    """
    # Ensure 30-min frequency — fills any missing timestamp slots with NaN
    load = load.resample("30min").mean()

    n_missing = load.isna().sum()
    if n_missing > 0:
        print(f"  Filling {n_missing} missing values via linear interpolation...")
        load = load.interpolate(method="linear", limit=4)  # max 2h gap

    # Outlier detection: flag values beyond 3 std of a 7-day rolling window
    roll_mean = load.rolling(window=336, center=True, min_periods=48).mean()
    roll_std  = load.rolling(window=336, center=True, min_periods=48).std()
    outliers  = (load - roll_mean).abs() > 3 * roll_std
    n_out = outliers.sum()
    if n_out > 0:
        print(f"  Replacing {n_out} outliers with rolling mean...")
        load[outliers] = roll_mean[outliers]

    # Drop any remaining NaN rows at start/end (rolling window edges)
    load = load.dropna()
    print(f"  Clean series: {len(load)} records from {load.index[0]} to {load.index[-1]}")
    return load


# ─────────────────────────────────────────
# STEP 3: Add temporal context features
# ─────────────────────────────────────────
def add_features(load: pd.Series, country: str = "DE_LU") -> pd.DataFrame:
    """
    Adds time-based context that helps LLMs reason about patterns.
    These become part of the prompt context (like temp/holiday in the energy sample data).

    Rolling window is 1 week (336 steps) rather than 24h so that:
      - The same weekday is always included in the reference baseline
      - Country-specific seasonal swings are captured automatically
      - No hardcoded season assumptions are needed

    Season labels are country-aware since peak seasons differ by region:
      - Northern/Central Europe (DE, FR, PL...): winter-peaking grid
      - Southern Europe (ES, IT, GR...):         summer-peaking grid
      - Nordic (NO, SE, FI...):                  extreme winter-peaking
      - Southern Hemisphere (AU...):             reversed seasons
    """
    df = load.to_frame()
    df["hour"]        = df.index.hour
    df["day_of_week"] = df.index.dayofweek   # 0=Mon, 6=Sun
    df["month"]       = df.index.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    # --- Rolling window: 1 week = 336 x 30min slots ---
    # This captures weekly cycles (weekday vs weekend) which dominate
    # load patterns in every country, regardless of season definition.
    # It also adapts to seasonal drift automatically — no season label needed
    # for the threshold calculation itself.
    ROLL = 336  # 7 days × 48 slots/day
    df["rolling_mean_1w"] = load.rolling(ROLL, min_periods=48).mean().values
    df["rolling_std_1w"]  = load.rolling(ROLL, min_periods=48).std().values

    # Keep 24h stats too — useful as a secondary prompt feature
    df["rolling_mean_24h"] = load.rolling(48, min_periods=24).mean().values
    df["rolling_std_24h"]  = load.rolling(48, min_periods=24).std().values

    # --- Country-aware season labels (for prompt context only) ---
    # These do NOT affect the threshold calculation.
    # They tell the LLM what season it is in human terms.
    SOUTHERN_HEMISPHERE = {"AU", "NZ", "ZA"}
    SUMMER_PEAKING      = {"ES", "PT", "GR", "IT", "CY", "MT"}  # AC-dominant
    NORDIC              = {"NO_1","NO_2","NO_3","NO_4","NO_5","SE_1","SE_2",
                           "SE_3","SE_4","FI","DK_1","DK_2"}

    cc = country.split("_")[0]  # strip bidding zone suffix, e.g. DE_LU → DE

    if cc in SOUTHERN_HEMISPHERE:
        # Reversed calendar: Jun/Jul/Aug = winter, Dec/Jan/Feb = summer
        season_map = lambda m: (
            2 if m in [12,1,2] else 3 if m in [3,4,5]
            else 0 if m in [6,7,8] else 1
        )
        peak_season_note = "Southern Hemisphere (summer = Dec–Feb)"
    elif cc in SUMMER_PEAKING:
        # Mediterranean: summer is the high-demand season
        season_map = lambda m: (
            0 if m in [12,1,2] else 1 if m in [3,4,5]
            else 2 if m in [6,7,8] else 3
        )
        peak_season_note = "Summer-peaking grid (AC-dominant)"
    else:
        # Default: Northern/Central Europe + Nordic
        # Winter is the high-demand season
        season_map = lambda m: (
            0 if m in [12,1,2] else 1 if m in [3,4,5]
            else 2 if m in [6,7,8] else 3
        )
        peak_season_note = "Winter-peaking grid (heating-dominant)"

    season_names = ["Winter", "Spring", "Summer", "Autumn"]
    df["season"]           = df["month"].map(season_map)
    df["season_name"]      = df["season"].map(lambda s: season_names[s])
    df["peak_season_note"] = peak_season_note  # stored for prompt context

    return df.dropna()


# ─────────────────────────────────────────
# STEP 4: Label extreme demand events
# ─────────────────────────────────────────
def label_extreme_events(df: pd.DataFrame, std_threshold=2.0) -> pd.DataFrame:
    """
    Labels each 30-min slot as extreme (1) or normal (0).

    Definition: extreme = load > rolling_mean + std_threshold * rolling_std
    This mirrors your extreme weather labels (storm/flood yes/no).

    For the LLM prompt, we ask: "Will there be an extreme demand event
    in the next 24 hours?" — binary classification, same as your extreme weather task.
    """
    df = df.copy()
    # Use 1-week rolling stats (captures weekly cycle + seasonal drift)
    # rather than 24h stats (too short, misses weekday/weekend pattern)
    threshold = df["rolling_mean_1w"] + std_threshold * df["rolling_std_1w"]
    df["is_extreme"] = (df["load_mw"] > threshold).astype(int)

    # Also record how extreme (magnitude above threshold in MW)
    df["extreme_magnitude_mw"] = (df["load_mw"] - threshold).clip(lower=0)

    n_extreme = df["is_extreme"].sum()
    pct = n_extreme / len(df) * 100
    print(f"  Extreme events: {n_extreme}/{len(df)} slots ({pct:.1f}%)")
    return df


# ─────────────────────────────────────────
# STEP 5: Build prompt-ready JSON examples
# ─────────────────────────────────────────
def build_forecast_examples(df: pd.DataFrame, history_len=48, pred_len=48, stride=48) -> list:
    """
    Converts the DataFrame into prompt-ready examples for your runner.py.

    Format mirrors your existing weather dataset structure:
    {
        "key": "energy_DE_2023-01-15_00:00",
        "task": "energy_forecast",
        "history": [91.6, 87.8, 75.2, ...],        # 48 values = 24h of history
        "target": [88.1, 82.3, 71.0, ...],          # 48 values = next 24h to predict
        "context": {                                  # metadata for the prompt
            "start_timestamp": "2023-01-15 00:00 UTC",
            "region": "DE_LU",
            "hour_of_day": 0,
            "day_of_week": "Sunday",
            "month": 1,
            "season": "Winter",
            "is_weekend": 1,
            "rolling_mean_mw": 45230.1,
            "rolling_std_mw": 3210.4
        }
    }
    """
    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    seasons = ["Winter","Spring","Summer","Autumn"]
    examples = []

    for i in range(0, len(df) - history_len - pred_len, stride):
        hist_df  = df.iloc[i : i + history_len]
        pred_df  = df.iloc[i + history_len : i + history_len + pred_len]
        start_ts = hist_df.index[0]

        ex = {
            "key": f"energy_{COUNTRY}_{start_ts.strftime('%Y-%m-%d_%H:%M')}",
            "task": "energy_forecast",
            "history": hist_df["load_mw"].round(2).tolist(),
            "target": pred_df["load_mw"].round(2).tolist(),
            "context": {
                "start_timestamp": str(start_ts),
                "region": COUNTRY,
                "hour_of_day": int(hist_df.iloc[-1]["hour"]),
                "day_of_week": days[int(hist_df.iloc[-1]["day_of_week"])],
                "month": int(hist_df.iloc[-1]["month"]),
                "season": seasons[int(hist_df.iloc[-1]["season"])],
                "is_weekend": int(hist_df.iloc[-1]["is_weekend"]),
                "rolling_mean_mw": round(float(hist_df["rolling_mean_24h"].mean()), 2),
                "rolling_std_mw": round(float(hist_df["rolling_std_24h"].mean()), 2),
            }
        }
        examples.append(ex)

    return examples


def build_extreme_examples(df: pd.DataFrame, history_len=48, stride=48) -> list:
    """
    Builds extreme event detection examples.

    Each example gives the LLM 24h of history and asks:
    "Will there be an extreme demand event in the NEXT 24 hours?"

    Mirrors your extreme weather detection task exactly.
    Ground truth: has_extreme (bool) + peak_magnitude (MW above threshold)
    """
    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    seasons = ["Winter","Spring","Summer","Autumn"]
    examples = []

    for i in range(0, len(df) - history_len * 2, stride):
        hist_df   = df.iloc[i : i + history_len]
        future_df = df.iloc[i + history_len : i + history_len * 2]
        start_ts  = hist_df.index[0]

        has_extreme   = bool(future_df["is_extreme"].any())
        peak_mag      = float(future_df["extreme_magnitude_mw"].max())
        n_extreme_pts = int(future_df["is_extreme"].sum())

        ex = {
            "key": f"energy_extreme_{COUNTRY}_{start_ts.strftime('%Y-%m-%d_%H:%M')}",
            "task": "energy_extreme",
            "history": hist_df["load_mw"].round(2).tolist(),
            "ground_truth": {
                "has_extreme": has_extreme,              # boolean — like your storm detection
                "n_extreme_slots": n_extreme_pts,         # how many 30-min slots are extreme
                "peak_magnitude_mw": round(peak_mag, 2), # how far above threshold (MW)
            },
            "context": {
                "start_timestamp": str(start_ts),
                "region": COUNTRY,
                "day_of_week": days[int(hist_df.iloc[-1]["day_of_week"])],
                "season": seasons[int(hist_df.iloc[-1]["season"])],
                "is_weekend": int(hist_df.iloc[-1]["is_weekend"]),
                "rolling_mean_mw": round(float(hist_df["rolling_mean_1w"].mean()), 2),
                "rolling_std_mw":  round(float(hist_df["rolling_std_1w"].mean()), 2),
                "peak_season_note": hist_df["peak_season_note"].iloc[-1],
            }
        }
        examples.append(ex)

    return examples


# ─────────────────────────────────────────
# STEP 6: Build LLM prompt strings
# ─────────────────────────────────────────
def make_forecast_prompt(example: dict) -> str:
    """Mirrors the prompt structure from your existing prompts.py"""
    ctx = example["context"]
    hist_str = ", ".join(str(v) for v in example["history"])
    return (
        f"The following is electricity load data (MW) recorded at 30-minute intervals "
        f"in the {ctx['region']} bidding zone.\n"
        f"Region: {ctx['region']}. Season: {ctx['season']}. "
        f"Day: {ctx['day_of_week']}. Is weekend: {'Yes' if ctx['is_weekend'] else 'No'}.\n"
        f"Historical average load: {ctx['rolling_mean_mw']} MW.\n\n"
        f"Historical load (last 24 hours, 48 values):\n{hist_str}\n\n"
        f"Task: Predict the electricity load (MW) for each of the next 48 time steps "
        f"(next 24 hours). Output exactly 48 comma-separated numbers."
    )


def make_extreme_prompt(example: dict) -> str:
    """Mirrors your extreme weather prompt structure"""
    ctx = example["context"]
    hist_str = ", ".join(str(v) for v in example["history"])
    threshold = ctx["rolling_mean_mw"] + EXTREME_STD * ctx["rolling_std_mw"]
    return (
        f"The following is electricity load data (MW) in the {ctx['region']} bidding zone.\n"
        f"Season: {ctx['season']}. Day: {ctx['day_of_week']}. "
        f"Is weekend: {'Yes' if ctx['is_weekend'] else 'No'}.\n"
        f"Normal load range: {ctx['rolling_mean_mw']:.0f} ± {ctx['rolling_std_mw']:.0f} MW. "
        f"Extreme threshold: {threshold:.0f} MW.\n\n"
        f"Historical load (last 24 hours, 48 values):\n{hist_str}\n\n"
        f"Task: Based on the historical load pattern, will there be an extreme demand event "
        f"(load exceeding {threshold:.0f} MW) in the NEXT 24 hours?\n"
        f"Answer with: YES or NO. If YES, also estimate the peak load in MW."
    )


# ─────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────
def run_pipeline():
    # 1. Download
    load_raw = download_load(API_KEY, COUNTRY, START, END)

    # 2. Clean
    load_clean = clean_load(load_raw)

    # 3. Feature engineering
    df = add_features(load_clean, country=COUNTRY)

    # 4. Label extremes
    df = label_extreme_events(df, std_threshold=EXTREME_STD)

    # 5. Build examples
    print("\nBuilding forecast examples...")
    forecast_examples = build_forecast_examples(df, HISTORY_LEN, PRED_LEN)
    print(f"  {len(forecast_examples)} forecast examples created")

    print("Building extreme event examples...")
    extreme_examples = build_extreme_examples(df, HISTORY_LEN)
    n_pos = sum(1 for e in extreme_examples if e["ground_truth"]["has_extreme"])
    n_neg = len(extreme_examples) - n_pos
    print(f"  Raw: {len(extreme_examples)} examples ({n_pos} positive, {n_neg} negative)")

    # --- Balance and cap at TARGET_TOTAL ---
    if BALANCE:
        import random
        random.seed(42)  # reproducible sampling
        pos = [e for e in extreme_examples if     e["ground_truth"]["has_extreme"]]
        neg = [e for e in extreme_examples if not e["ground_truth"]["has_extreme"]]

        if len(pos) == 0:
            print("  ⚠️  No positive examples found!")
            print("  Try: lower EXTREME_STD (e.g. 1.5→1.2), extend date range, or check data.")
        else:
            # Cap each class at half of TARGET_TOTAL
            n_each = min(TARGET_TOTAL // 2, len(pos), len(neg))
            pos_sampled = random.sample(pos, n_each)
            neg_sampled = random.sample(neg, n_each)
            extreme_examples = pos_sampled + neg_sampled
            random.shuffle(extreme_examples)
            print(f"  Balanced: {len(extreme_examples)} examples "
                  f"({n_each} positive, {n_each} negative)")

    # 6. Add prompts
    for ex in forecast_examples:
        ex["prompt"] = make_forecast_prompt(ex)
    for ex in extreme_examples:
        ex["prompt"] = make_extreme_prompt(ex)

    # 7. Save
    with open("energy_forecast_new.json", "w") as f:
        json.dump(forecast_examples, f, indent=2, default=str)
    with open("energy_extreme_new.json", "w") as f:
        json.dump(extreme_examples, f, indent=2, default=str)

    print("\n✅ Saved: energy_forecast_new.json, energy_extreme_new.json")
    print(f"   Forecast examples: {len(forecast_examples)}")
    print(f"   Extreme examples:  {len(extreme_examples)}")

    # Show one sample
    print("\n--- Sample Forecast Prompt ---")
    print(forecast_examples[0]["prompt"][:600])
    print("\n--- Sample Extreme Prompt ---")
    print(extreme_examples[0]["prompt"][:600])

    return forecast_examples, extreme_examples


if __name__ == "__main__":
    forecast_data, extreme_data = run_pipeline()