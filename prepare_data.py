import json
import numpy as np

JSONL_PATH = "data/weather/10_test_data.jsonl"
SEQ_LEN = 6

FEATURE_ORDER = [
    "east_west_wind_speed_10m",
    "north_south_wind_speed_10m",
    "dewpoint_temperature_2m",
    "air_temperature_2m",
    "mean_sea_level_pressure",
    "surface_pressure",
    "total_precipitation",
]

def load_first_item(path):
    with open(path, "r") as f:
        line = f.readline()
    return json.loads(line)

def build_XY(item, seq_len=6):
    obs = item["obs_json"]
    gt = item["ground_truth"]

    x_times_all = sorted([k for k in obs.keys() if k != "question"])
    x_times = x_times_all[-seq_len:]  # ambil last seq_len
    y_times = sorted(gt.keys())

    X = np.array([[obs[t][f] for f in FEATURE_ORDER] for t in x_times], dtype=np.float32)
    Y = np.array([[gt[t][f] for f in FEATURE_ORDER] for t in y_times], dtype=np.float32)

    area = obs[x_times_all[0]]["area"]
    return X, Y, x_times, y_times, area

if __name__ == "__main__":
    item = load_first_item(JSONL_PATH)

    print("Top-level keys:", list(item.keys()))
    print("\nobservation (snippet):", item["observation"][:200], "...")
    print("result (snippet):", item["result"][:200], "...")

    obs_keys = list(item["obs_json"].keys())
    gt_keys = list(item["ground_truth"].keys())
    print("\nobs_json keys (timestamps + question):", obs_keys)
    print("ground_truth keys (future timestamps):", gt_keys)

    X, Y, x_times, y_times, area = build_XY(item, seq_len=SEQ_LEN)
    print("\nArea:", area)
    print("X shape:", X.shape, " (seq_len, D)")
    print("Y shape:", Y.shape, " (H, D)")
    print("X times:", x_times)
    print("Y times:", y_times)

    print("\nX (first 2 rows):\n", X[:2])
    print("\nY:\n", Y)

