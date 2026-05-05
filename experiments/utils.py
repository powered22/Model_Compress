# utils.py
from datetime import datetime
import csv
import os
import sys
import json, ast, re

from typing import Any, Dict, Optional, Literal,List
from ollama import AsyncClient as OllamaClient  # <-- requires `pip install ollama`
import logging
logger_initialized = {}

OPS = ['encode', 'decode']
MODES = Literal['easy', 'hard']
OLLAMA_MODEL = os.environ.get("WEATHER_OLLAMA_MODEL", "mistral")

CSV_WEATHER = ['Challenge', 'Prompt', 'Ground Truth', 'LLM Result', 'LLM Result (Clean)', 'Levenshtein Distance',
               'Normalized Levenshtein Distance', 'LCS', 'NLCS', 'MAE Penalised', 'MAE Matched','Timestamp Coverage %','COVERAGE' , 'BertScore', 'CosineSimilarity', 'BLEU', 'ROUGE-L', 'STRICT', 'LENIENT'
                ,'Model','Operation', 'Mode', 'Prompting', 'LLM Result in sec']
CSV_EXTREME_WEATHER = ['Challenge', 'Prompt', 'LLM Result',  'PRED (yes or no)', 'PRED TYPE', 'GROUND TRUTH (yes or no)','GROUND TRUTH TYPE','STRICT','LENIENT','CONSISTENCY'
                       'Model', 'Mode', 'Prompting', 'LLM Result in sec']

_ALLOWED_EVENT_TYPES = {
    "Hail",
    "Thunderstorm Wind",
    "Flash Flood",
    "Tornado",
    "Lightning",
    "Flood",
    "Funnel Cloud",
    "NA",
}

_YES_PAT = re.compile(r"\b(yes|y|true)\b", re.IGNORECASE)
_NO_PAT  = re.compile(r"\b(no|n|false)\b", re.IGNORECASE)

class Logger:
    def __init__(self, title: str, model: str, columns: list):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = './logs/'
        os.makedirs(log_dir, exist_ok=True)
        self.csv_file_name = f'{log_dir}{title}_{model}_experiment_log_{timestamp}.csv'
        self.csv_file = open(self.csv_file_name, mode='w', newline='', encoding='utf-8-sig')
        self.csv_writer = csv.writer(self.csv_file)
        self.columns = columns
        self.csv_writer.writerow(columns)
        self.csv_file.close()

    def log(self, row: list):
        self.csv_file = open(self.csv_file_name, mode='a', newline='', encoding='utf-8-sig')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(row)
        self.csv_file.close()

    def sort(self, field: str = 'pos_id'):
        # Read CSV file
        with open(self.csv_file_name, mode='r', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            data = sorted(reader, key=lambda row: int(row[field]))  # Sort by column

        # Write back to the same file
        with open(self.csv_file_name, mode='w', newline='', encoding='utf-8-sig') as file:
            writer = csv.DictWriter(file, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(data)

        self.csv_file.close()

    def end(self):
        self.csv_file.close()


def get_logging(name='weather_logger',
               log_dir='results_log',
               log_prefix='run',
               rank=0,
               master_only=True,
               timestamp=None):
    """
    Unified logger for both single and multi-rank logging.
    """

    # full name per rank
    full_name = f"{name}:rank{rank}"

    if full_name in logger_initialized:
        return logging.getLogger(full_name)

    logger = logging.getLogger(full_name)

    # level per rank
    if rank == 0:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING if master_only else logging.INFO

    logger.setLevel(log_level)
    formatter = logging.Formatter(f'[rank{rank}] [%(asctime)s] [%(levelname)s] - %(message)s',
                                   "%Y-%m-%d %H:%M:%S")

    # log to file (only rank 0)
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(log_dir, f'{log_prefix}_{timestamp}.log')
        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    # log to stdout
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)
    logger.addHandler(stream_handler)

    logger_initialized[full_name] = True
    return logger

WEATHER_FIELDS = [
    "east_west_wind_speed_10m",
    "north_south_wind_speed_10m",
    "dewpoint_temperature_2m",
    "air_temperature_2m",
    "mean_sea_level_pressure",
    "surface_pressure",
    "total_precipitation",
    "latitude",
    "longitude",
    "area",
]


def _build_extraction_prompt(raw_text: str) -> str:
    schema_example = (
        "{"
        "'2012-12-22 13:00:00': {"
        "'east_west_wind_speed_10m': 0.37, "
        "'north_south_wind_speed_10m': 0.23, "
        "'dewpoint_temperature_2m': 15.72, "
        "'air_temperature_2m': 16.59, "
        "'mean_sea_level_pressure': 1012.15, "
        "'surface_pressure': 820.79, "
        "'total_precipitation': 0.005, "
        "'latitude': -4.5, "
        "'longitude': 140.5, "
        "'area': 'Pegunungan Bintang, Highland Papua'"
        "}"
        "}"
    )

    return f"""
You are a weather data extractor.

Your task:
- Convert the following narrative into a JSON object (not an array).
- Top-level keys: timestamps formatted as **YYYY-MM-DD HH:MM:SS**.
  *If only hour/minute are available, fill seconds with "00".*
- For each timestamp, include ALL of the following fields (use null if missing):
  {WEATHER_FIELDS}
- Data types:
  - All numeric values as numbers (floats), not strings.
  - latitude/longitude: float or null.
  - area: string or null.
- Do not add any extra fields.
- Return ONLY a JSON object — no explanations, no comments, no code fences.

Example structure (not actual required values):
{schema_example}

Narrative to extract:
---
{raw_text}
---
Return ONLY the JSON object as specified above.
""".strip()


async def _ollama_chat(prompt: str, model: Optional[str] = None) -> str:
    """
    Call Ollama via the official python-ollama client (async).
    """
    messages = [{"role": "user", "content": prompt}]
    response = await OllamaClient().chat(model=model or OLLAMA_MODEL, messages=messages,format="json",options={"temperature": 0.1})
    return str(response["message"]["content"])


def _coerce_numeric(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", ""))
        except Exception:
            return None
    return None


def canon_ts(ts: Any) -> Optional[str]:
    """
    Normalisasikan timestamp longgar ke format kanonik: 'YYYY-MM-DD HH:MM:SS'.
    - Izinkan 'YYYY/MM/DD' -> 'YYYY-MM-DD'
    - Izinkan separator jam '.' -> ':' (01.00[:SS] -> 01:00[:SS])
    - Tambahkan detik = '00' jika hilang.
    - Buang karakter nyasar di tepi: quotes, braces, dsb.
    Return None jika tidak bisa dinormalisasi.
    """
    if ts is None:
        return None
    s = str(ts).strip().strip(' "\'{}[](),')
    s = s.replace("T", " ")

    # ganti "HH.MM" atau "HH.MM.SS" -> "HH:MM(:SS)"
    m = re.search(r"(\d{1,2})[.:](\d{2})(?:[.:](\d{2}))?$", s)
    if m:
        hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
        s = re.sub(
            r"(\d{1,2})[.:](\d{2})(?:[.:](\d{2}))?$",
            f"{hh:02d}:{mm:02d}:{ss:02d}",
            s,
        )

    # tanggal 'YYYY/MM/DD' -> 'YYYY-MM-DD'
    s = re.sub(
        r"(\d{4})/(\d{1,2})/(\d{1,2})",
        lambda m: f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}",
        s,
    )

    # jika belum ada detik, tambahkan
    if re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$", s):
        s += ":00"

    # valid akhir
    if not re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", s):
        return None
    return s


def rekey_timeseries(
    obj: Dict[str, Dict[str, Any]],
    prefer_numeric_fields: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Rekey dict timeseries ke kunci timestamp kanonik.
    Jika 2 kunci menjadi sama setelah normalisasi, pilih baris
    dengan jumlah nilai numerik terisi lebih banyak berdasarkan `prefer_numeric_fields`.
    """
    if not isinstance(obj, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    prefer = prefer_numeric_fields or [
        "east_west_wind_speed_10m",
        "north_south_wind_speed_10m",
        "dewpoint_temperature_2m",
        "air_temperature_2m",
        "mean_sea_level_pressure",
        "surface_pressure",
        "total_precipitation",
    ]
    def _num_count(row: Dict[str, Any]) -> int:
        return sum(1 for k in prefer if isinstance(row.get(k), (int, float)))
    for k, v in obj.items():
        if not isinstance(v, dict):
            continue
        nk = canon_ts(k)
        if not nk:
            continue
        if nk in out:
            out[nk] = v if _num_count(v) >= _num_count(out[nk]) else out[nk]
        else:
            out[nk] = v
    return out


def _sanitize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for k in WEATHER_FIELDS:
        val = row.get(k, None)
        if k == "area":
            cleaned[k] = (val if (val is None or isinstance(val, str)) else str(val))
        else:
            cleaned[k] = _coerce_numeric(val)
    return cleaned

def _json_from_model_output(text: str) -> dict:
    """
    Try hard to recover a JSON object from model output.
    Never raises; returns {} on failure.
    """
    if not isinstance(text, str):
        return {}

    s = text.strip()

    # strip code fences ```...```
    if s.startswith("```"):
        fence_end = s.rfind("```")
        if fence_end > 0:
            s = s[3:fence_end].strip()
            if s.lower().startswith("json"):
                s = s[4:].strip()

    # crop to outermost {...}
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1 and j > i:
        s = s[i:j+1]

    # 1) first attempt: strict JSON
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    # 2) quick repairs: smart quotes, trailing commas, NaN/Infinity
    repaired = (s
        .replace("“", '"').replace("”", '"')
        .replace("’", "'").replace("‘", "'")
    )
    # remove trailing commas before } or ]
    repaired = re.sub(r",\s*(?=[}\]])", "", repaired)
    # replace NaN/Infinity with null
    repaired = re.sub(r"\bNaN\b|\bInfinity\b|\b-Infinity\b", "null", repaired)

    try:
        obj = json.loads(repaired)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    # 3) fallback: parse as Python literal then coerce to JSONable
    try:
        obj = ast.literal_eval(repaired)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

async def extract_weather_json_async(text: str) -> Dict[str, Dict[str, Any]]:
    if not isinstance(text, str) or not text.strip():
        return {}
    prompt = _build_extraction_prompt(text)
    llm_text = await _ollama_chat(prompt)  # panggilan ke OllamaClient().chat
    obj = _json_from_model_output(llm_text)
    if not isinstance(obj, dict) or not obj:
        return {}
    out = {}
    for ts, row in obj.items():
        if isinstance(row, dict):
            out[str(ts)] = _sanitize_row(row)

    out = rekey_timeseries(out)
    return out


def _normalize_event_type(s: str) -> str:
    s = (s or "").strip()
    # Common normalization variants
    s_low = s.lower()
    mapping = {
        "tstm wind": "Thunderstorm Wind",
        "thunderstormwind": "Thunderstorm Wind",
        "thunderstorm wind": "Thunderstorm Wind",
        "flashflood": "Flash Flood",
        "flash flood": "Flash Flood",
        "funnelcloud": "Funnel Cloud",
        "funnel cloud": "Funnel Cloud",
    }
    if s in _ALLOWED_EVENT_TYPES:
        return s
    if s_low in mapping:
        return mapping[s_low]
    # Title-case fallback
    s2 = " ".join(w.capitalize() for w in s.split())
    return s2 if s2 in _ALLOWED_EVENT_TYPES else "NA"

def parse_extreme_response(llm_response: str) -> Dict[str, Any]:
    """
    Output schema (recommended):
      {
        "has_extreme_weather": bool,
        "event_type": "Hail|Thunderstorm Wind|Flash Flood|Tornado|Lightning|Flood|Funnel Cloud|NA",
        "rule": "optional free-text decision rule"
      }

    This parser tries:
      (1) strict JSON object in response
      (2) JSON-ish substring
      (3) fallback regex on free text
    """
    llm_response = llm_response or ""
    txt = llm_response.strip()

    # (1) Try full JSON
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            has = bool(obj.get("has_extreme_weather", obj.get("extreme", obj.get("yes"))))
            et  = _normalize_event_type(str(obj.get("event_type", obj.get("type", "NA"))))
            rule = obj.get("rule", "")
            return {"has_extreme_weather": has, "event_type": et, "rule": rule}
    except Exception:
        pass

    # (2) Try JSON substring {...}
    try:
        m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
        if m:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                has = bool(obj.get("has_extreme_weather", obj.get("extreme", obj.get("yes"))))
                et  = _normalize_event_type(str(obj.get("event_type", obj.get("type", "NA"))))
                rule = obj.get("rule", "")
                return {"has_extreme_weather": has, "event_type": et, "rule": rule}
    except Exception:
        pass

    # (3) Regex fallback
    # - Determine yes/no
    has: Optional[bool] = None
    if _YES_PAT.search(txt) and not _NO_PAT.search(txt):
        has = True
    elif _NO_PAT.search(txt):
        has = False
    else:
        has = False  # default conservative

    # - Determine event type by searching allowed strings
    #   Prefer exact labels if present
    found_type = None
    for t in sorted(_ALLOWED_EVENT_TYPES, key=len, reverse=True):
        if t != "NA" and re.search(rf"\b{re.escape(t)}\b", txt, flags=re.IGNORECASE):
            found_type = t
            break

    et = _normalize_event_type(found_type or "NA")
    if not has:
        et = "NA"

    # - Optional: grab a “rule:” line if present
    rule = ""
    rm = re.search(r"(decision rule|rule)\s*[:\-]\s*(.+)", txt, flags=re.IGNORECASE)
    if rm:
        rule = rm.group(2).strip()

    return {"has_extreme_weather": has, "event_type": et, "rule": rule}