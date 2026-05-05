from datetime import datetime, timedelta
from typing import Dict, Any
from experiments.metrics import Metrics

TARGET_FIELDS = [
    "east_west_wind_speed_10m",
    "north_south_wind_speed_10m",
    "dewpoint_temperature_2m",
    "air_temperature_2m",
    "mean_sea_level_pressure",
    "surface_pressure",
    "total_precipitation",
]


def persistence_predict(obs_json: dict, horizon: int) -> Dict[str, Dict[str, Any]]:
    """
    Persistence baseline: repeat the last observed timestep for H future steps.

    Args:
        obs_json: your obs_json dict (keys = timestamps + "question")
        horizon:  number of hours to forecast (parsed from the question)

    Returns:
        prediction dict in the same format as ground_truth
        e.g. {"2012-12-22 13:00:00": {"air_temperature_2m": 16.55, ...}, ...}
    """
    # Step 1: get all observation timestamps, sorted
    obs_timestamps = sorted([k for k in obs_json.keys() if k != "question"])

    # Step 2: grab the last observed row
    last_ts = obs_timestamps[-1]  # "2012-12-22 12:00:00"
    last_row = obs_json[last_ts]

    # Step 3: copy only the 7 numeric forecast fields (drop lat/lon/area)
    last_values = {f: last_row.get(f) for f in TARGET_FIELDS}

    # Step 4: generate future timestamps by incrementing 1 hour at a time
    last_dt = datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S")
    prediction = {}
    for h in range(1, horizon + 1):
        future_dt = last_dt + timedelta(hours=h)
        future_ts = future_dt.strftime("%Y-%m-%d %H:%M:%S")
        prediction[future_ts] = dict(last_values)  # copy, not reference

    return prediction


def run_persistence_baseline(data_list: list) -> dict:
    """
    Run persistence baseline over your full dataset and return averaged metrics.

    Args:
        data_list: list of dicts, each with keys: obs_json, ground_truth

    Returns:
        dict of averaged metrics (same keys as your Metrics.get() output)
    """
    from experiments.utils import parse_horizon_from_question

    metrics = Metrics()

    for datum in data_list:
        obs_json = datum["obs_json"]
        ground_truth = datum["ground_truth"]

        # Parse how many hours to forecast from the question string
        question = obs_json.get("question", "")
        horizon = parse_horizon_from_question(question)  # already in utils.py

        # Generate persistence prediction
        pred = persistence_predict(obs_json, horizon)

        # Evaluate using your existing metrics pipeline
        # mode_json=True because pred is already a dict (no extraction needed)
        scores = metrics.update_weather(
            json_llm_response=pred,
            sentence_llm_response="",  # no text — persistence has no text output
            ground_truth=ground_truth,
            result_sentence="",
            mode_json=True
        )

    return metrics.get(subject="weather")


# ── Quick demo with your sample data ──────────────────────────────────────────
if __name__ == "__main__":
    import json

    sample = {
        "obs_json": {
            "2012-12-22 06:00:00": {"east_west_wind_speed_10m": -0.35, "north_south_wind_speed_10m": 0.23,
                                    "dewpoint_temperature_2m": 18.31, "air_temperature_2m": 19.86,
                                    "mean_sea_level_pressure": 1008.92, "surface_pressure": 818.56,
                                    "total_precipitation": 0.0011, "area": "Pegunungan Bintang"},
            # ... (other hours) ...
            "2012-12-22 12:00:00": {"east_west_wind_speed_10m": 0.65, "north_south_wind_speed_10m": 0.40,
                                    "dewpoint_temperature_2m": 15.83, "air_temperature_2m": 16.55,
                                    "mean_sea_level_pressure": 1011.97, "surface_pressure": 820.71,
                                    "total_precipitation": 0.0041, "area": "Pegunungan Bintang"},
            "question": "Please describe weather observation in the next 3 hours using the same format as in the question."
        }
    }

    pred = persistence_predict(sample["obs_json"], horizon=3)
    print("Persistence prediction:")
    for ts, row in pred.items():
        print(f"  {ts}: temp={row['air_temperature_2m']}, pressure={row['mean_sea_level_pressure']}")

    # Expected output:
    # 2012-12-22 13:00:00: temp=16.55, pressure=1011.97   ← same as 12:00
    # 2012-12-22 14:00:00: temp=16.55, pressure=1011.97   ← same as 12:00
    # 2012-12-22 15:00:00: temp=16.55, pressure=1011.97   ← same as 12:00