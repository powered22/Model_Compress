import pandas as pd
import asyncio
import random
import logging
from pathlib import Path
#from experiments import utils
#from experiments.experiments import WeatherExperiment
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("results_log/weather_main.log",mode='w'),
        logging.StreamHandler()
    ]
)

semaphore = asyncio.Semaphore(5)

BLOCK_SIZE = 10

def convert_dataset(df, kelvin_cols=["d2m", "t2m"], pa=["msl", "sp"]):
    for col in kelvin_cols:
        if col in df.columns:
            df[col] = df[col] - 273.15
    for col in pa:
        if col in df.columns:
            df[col] = df[col] / 100
    return df

def format_row_as_text(row, timestamp_col='valid_time'):
    try:
        time_str = pd.to_datetime(row[timestamp_col]).strftime('%Y-%m-%d %H:%M')
        location = row.get('area', 'Unknown Location')

        wind_u_10m = f"{row.get('u10', 0):.2f}"
        wind_v_10m = f"{row.get('v10', 0):.2f}"
        dewpoint_temp_2m = f"{row.get('d2m', 0):.2f}"
        temperature = f"{row.get('t2m', 0):.2f}"
        pressure_msl = f"{row.get('msl', 0):.2f}"
        surface_pressure = f"{row.get('sp', 0):.2f}"
        total_precipitation = f"{row.get('tp', 0):.4f}"

        return (
            f"At {time_str}, in {location}, the weather observations were recorded as follows:\n"
            f"East – west wind speed at 10 meters above ground: {wind_u_10m} m/s, "
            f"North – south wind speed at 10 meters above ground: {wind_v_10m} m/s. "
            f"Dewpoint Temperature: {dewpoint_temp_2m} °C. Temperature: {temperature} °C. "
            f"Pressure Mean Sea Level: {pressure_msl} hPa. Surface Pressure: {surface_pressure} hPa. "
            f"Total Precipitation: {total_precipitation} meters"
        )
    except Exception as e:
        return f"[Error formatting row: {e}]"

def generate_summaries(data, output_path):
    df = convert_dataset(data)
    df['valid_time'] = pd.to_datetime(df['valid_time'])
    df = df.sort_values('valid_time').reset_index(drop=True)
    total_rows = len(df)
    blocks = total_rows // BLOCK_SIZE

    # logging.info(f"Generating summaries for {total_rows} rows, {blocks} blocks")
    # print(f"{df.head()}")

    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(blocks):
            try:
                block_df = df.iloc[i * BLOCK_SIZE: (i + 1) * BLOCK_SIZE]
                obs_len = random.randint(5, BLOCK_SIZE - 2)
                target_len = BLOCK_SIZE - obs_len

                obs_df = block_df.iloc[:obs_len]
                target_df = block_df.iloc[obs_len:]

                obs_text = "\n".join(format_row_as_text(row) for _, row in obs_df.iterrows())
                result_text = "\n".join(format_row_as_text(row) for _, row in target_df.iterrows())

                ground_truth_df = target_df.copy()
                ground_truth_df["valid_time"] = ground_truth_df["valid_time"].dt.strftime("%Y-%m-%d %H:%M:%S")

                ground_truth_dicts = ground_truth_df[[
                    "valid_time", "u10", "v10", "d2m",
                    "t2m", "msl", "sp",
                    "tp", "latitude", "longitude", "area"
                ]].to_dict(orient="records")

                entry = {
                    "observation": f"{obs_text.strip()}\n\nPlease describe weather observation in the next {target_len} hours.",
                    "result": result_text.replace("\n", " "),
                    "ground_truth": ground_truth_dicts
                }

                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")

            except Exception as e:
                logging.error(f"Block {i + 1}: Error formatting summary: {e}")


if __name__ == "__main__":
    try:
        logging.basicConfig(level=logging.INFO)

        input_dir = Path("era5_data/ready_data")
        temp_output_dir = Path("era5_data/summaries_temp")
        final_output_file = Path("data/weather/all_combine_weather.jsonl")

        ##to combine convert from .zip to .jsonl
        # summary_files = []
        #
        # csv_count = len(list(input_dir.glob("*.csv")))
        # logging.info(f"CSV count: {csv_count}")
        #
        # # Step 1: Generate summary JSONL per CSV
        # for csv_file in input_dir.glob("*.csv"):
        #     try:
        #         logging.info(f"Processing: {csv_file.name}")
        #         df = pd.read_csv(csv_file)
        #         base_name = csv_file.stem
        #         temp_jsonl_path = temp_output_dir / f"{base_name}.jsonl"
        #         generate_summaries(df, temp_jsonl_path)
        #         summary_files.append(temp_jsonl_path)
        #     except Exception as e:
        #         logging.error(f"Error processing {csv_file.name}: {e}")
        #
        # #Step 2: Combine all .jsonl into one
        # with open(final_output_file, "w", encoding="utf-8") as outfile:
        #     for summary_file in summary_files:
        #         with open(summary_file, "r", encoding="utf-8") as infile:
        #             for line in infile:
        #                 outfile.write(line)

        ##just to check the final all_combine_weather.jsonl
        # with open(final_output_file, 'r', encoding='utf-8') as f:
        #     for i, line in enumerate(f, start=1):
        #         if i == 10517:
        #             data = json.loads(line)
        #             print(data['observation'])
        #             print(data['result'])
        #             print(data['ground_truth'])
        #             break

        # logging.info(f"✅ All summaries combined into: {final_output_file}")

        #reading the .jsonl specific the ground truth

        # target_line = 15 # Change to your desired line number
        #
        # with open("data/output.jsonl", "r", encoding="utf-8") as f:
        #     for line_number, line in enumerate(f, start=1):
        #         if line_number == target_line:
        #             if isinstance(line, str):  # Ensure it's a string before parsing
        #                 try:
        #                     data = json.loads(line.strip())
        #                     ground_truth = data.get("ground_truth", [])
        #                     print(f"[Line {line_number}] Ground truth:")
        #                     for item in ground_truth:
        #                         print(item)
        #                 except json.JSONDecodeError as e:
        #                     print(f"[Line {line_number}] Failed to decode JSON: {e}")
        #             else:
        #                 print(f"[Line {line_number}] Skipped: line is not a string (type={type(line)})")
        #             break
        # with open("data/output.jsonl", "r", encoding="utf-8") as f:
        #     line_count = sum(1 for _ in f)

        #print(f"The file has {line_count} lines.")
    except Exception:
        logging.exception("Something went wrong.")



