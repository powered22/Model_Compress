import json
import re
from typing import Tuple, Dict, Any, Optional

def _try_extract_json_block(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Try to find a JSON object in text and parse it.
    Returns (parsed_dict, error_or_none)
    """
    # simple heuristic: find first '{' .. '}' balanced block
    start = text.find('{')
    if start == -1:
        return None, "no_json_start"
    # try progressively larger slices until a valid JSON parse
    for end in range(text.rfind('}') + 1, start + 1, -1):
        try:
            candidate = text[start:end]
            parsed = json.loads(candidate)
            return parsed, None
        except Exception:
            continue
    return None, "json_parse_failed"

def _regex_numeric_fallback(text: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Deterministic fallback: extract the most likely numeric temperature (customize pattern for other variables).
    Returns (value, unit_or_none)
    """
    # Example pattern: capture "temperature: 12.3 C" or "temp 12.3C" or "12.3 C"
    pattern = re.compile(
        r"(?P<key>\btemp(?:erature)?\b)[^\d\-+]{0,12}(?P<value>-?\d+(?:\.\d+)?)(?:\s*(?P<unit>[CKF]))?",
        flags=re.IGNORECASE,
    )
    m = pattern.search(text)
    if m:
        val = float(m.group("value"))
        unit = (m.group("unit") or "C").upper()
        return val, unit
    # fallback: any standalone number
    m2 = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:C|K|F)?\b", text)
    if m2:
        return float(m2.group(1)), None
    return None, None

# python
async def parse_response_with_fallback(llm_text: str):
    """
    Try JSON extraction first using _try_extract_json_block.
    If that fails, run a deterministic regex fallback.
    Returns: (parsed_dict, extractor_ok: bool, info: Optional[str])
    """
    try:
        parsed, err = _try_extract_json_block(llm_text)
        if parsed:
            return parsed, True, None

        # No JSON start or JSON parse failed -> attempt deterministic fallback
        start_info = err or "no_json"
        pattern = re.compile(
            r"(?P<key>\b(temp|temperature|t)\b).{0,20}?(?P<value>-?\d+(\.\d+)?)\s*(?P<unit>C|K|F)?",
            re.IGNORECASE,
        )
        m = pattern.search(llm_text)
        if m:
            val = float(m.group("value"))
            unit = (m.group("unit") or "C").upper()
            fallback = {"temperature": {"value": val, "units": unit}}
            return fallback, True, f"fallback_from:{start_info}"

        # nothing found
        return {}, False, f"no_json_or_fallback:{start_info}"

    except Exception as e:
        return {}, False, f"exception:{repr(e)}"


async def extract_prediction(
    llm_text: str,
    expect_json: Optional[bool] = None,
    call_parser_model_fn=None,
) -> Dict[str, Any]:
    """
    Unified extractor:
      - If expect_json is True: prefer strict JSON parse (report extractor_ok False on parse fail).
      - If expect_json is False: try json block first (some sentence outputs still include JSON),
        then deterministic regex fallback.
      - If both fail and call_parser_model_fn provided, call it (ideally a different model) as a last resort.
    Returns dict with keys:
      - prediction: float | None
      - units: str | None
      - extractor_ok: bool
      - extractor_kind: 'json' | 'regex' | 'parser_model' | 'none'
      - extractor_info: short string for logs
    """
    # 1) JSON attempt (if expected or present)
    if expect_json is True or "{" in llm_text:
        parsed, err = _try_extract_json_block(llm_text)
        if parsed:
            # adapt to your schema: look for prediction field paths
            p = None
            units = None
            if isinstance(parsed, dict):
                if "prediction" in parsed:
                    p = parsed.get("prediction")
                    units = parsed.get("units")
                # fallback common keys
                elif "temperature" in parsed:
                    tp = parsed.get("temperature")
                    if isinstance(tp, dict):
                        p = tp.get("value")
                        units = tp.get("units")
                    elif isinstance(tp, (int, float)):
                        p = float(tp)
            try:
                p = float(p) if p is not None else None
            except Exception:
                p = None
            if p is not None:
                return {"prediction": p, "units": units, "extractor_ok": True, "extractor_kind": "json", "extractor_info": None}
            return {"prediction": None, "units": None, "extractor_ok": False, "extractor_kind": "json", "extractor_info": f"json_parsed_but_no_numeric:{err}"}

    # 2) Deterministic regex fallback
    val, unit = _regex_numeric_fallback(llm_text)
    if val is not None:
        return {"prediction": val, "units": unit, "extractor_ok": True, "extractor_kind": "regex", "extractor_info": None}

    # 3) Optional parser model (should be separate family/version to avoid correlated errors)
    if call_parser_model_fn:
        try:
            # call_parser_model_fn should accept text and return a dict like {"prediction":..., "units":...}
            parsed = await call_parser_model_fn(llm_text)
            if parsed and parsed.get("prediction") is not None:
                return {"prediction": float(parsed["prediction"]), "units": parsed.get("units"), "extractor_ok": True, "extractor_kind": "parser_model", "extractor_info": None}
            return {"prediction": None, "units": None, "extractor_ok": False, "extractor_kind": "parser_model", "extractor_info": "parser_model_no_value"}
        except Exception as e:
            return {"prediction": None, "units": None, "extractor_ok": False, "extractor_kind": "parser_model", "extractor_info": f"parser_exception:{e}"}

    # 4) Nothing found
    return {"prediction": None, "units": None, "extractor_ok": False, "extractor_kind": "none", "extractor_info": "no_extractor_matched"}