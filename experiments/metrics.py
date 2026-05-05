from typing import Dict
import json
import nltk
from difflib import SequenceMatcher
from nltk.corpus import stopwords

import json
from typing import Dict, Any, Tuple, Optional
from .utils import rekey_timeseries, canon_ts


sw = stopwords.words('english')

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from rouge_score import rouge_scorer
from bert_score import score as bertscore
from bert_score import BERTScorer
import ast
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)



TARGET_NUMERIC_FIELDS = [
    "east_west_wind_speed_10m",
    "north_south_wind_speed_10m",
    "dewpoint_temperature_2m",
    "air_temperature_2m",
    "mean_sea_level_pressure",
    "surface_pressure",
    "total_precipitation",
]


# Intersection Over Output
def IoO(encoding, enc_output, references, highlights):
    total = 0
    comparison = ''
    previous = False
    for token in enc_output:
        try:
            word = encoding.decode_single_token_bytes(token).decode().strip(" ")
        except UnicodeDecodeError:
            # print(encoding.decode_single_token_bytes(token))
            # exit()
            word = ""
        if word in sw:
            pass  # comparison = comparison + str(encoding.decode_single_token_bytes(token).decode())
        elif word in ["-", ".", ",", ":", "\"", ";", "?", "!", "|", "\\", "/", " ", "", "'s", "’s", "'re", "'t", "'ve",
                      "'m", "'ll"]:
            pass  # comparison = comparison + str(encoding.decode_single_token_bytes(token).decode())
        elif word in references:
            # comparison = comparison + highlights["yellow"] + "{" + str(encoding.decode_single_token_bytes(token).decode()) + "}"
            total = total + 1
            previous = True
        else:
            # comparison = comparison + str(encoding.decode_single_token_bytes(token).decode())
            previous = False
    return total / len(enc_output) if not len(enc_output) == 0 else 0


# Intersection Over Reference
def IoR(encoding, enc_references, output, highlights):
    total = 0
    comparison = ''
    previous = False
    for token in enc_references:
        try:
            encoding.decode_single_token_bytes(token).decode().strip(" ")
        except UnicodeDecodeError:
            continue
        word = encoding.decode_single_token_bytes(token).decode().strip(" ")

        if word in sw:
            comparison = comparison + str(encoding.decode_single_token_bytes(token).decode())
        elif word in ["-", ":", ".", ",", "\"", ";", "?", "!", "|", "\\", "/", " ", "", "'s", "’s", "'re", "'t", "'ve",
                      "'m", "'ll"]:
            comparison = comparison + str(encoding.decode_single_token_bytes(token).decode())
        elif word in output:
            comparison = comparison + highlights["yellow"] + "{" + str(
                encoding.decode_single_token_bytes(token).decode()) + "}"
            total = total + 1
            previous = True
        else:
            comparison = comparison + str(encoding.decode_single_token_bytes(token).decode())
            previous = False
    return total / len(enc_references) if not len(enc_references) == 0 else 0


# longest common substring
def longest_common_substring(resp: str, ans: str) -> float:
    match = SequenceMatcher(None, ''.join(resp), ''.join(ans)).find_longest_match()
    return len(''.join(ans)[match.b:match.b + match.size])

def norm_longest_common_substring(resp: str, ans: str) -> float:
    match = SequenceMatcher(None, ''.join(resp), ''.join(ans)).find_longest_match()
    return len(''.join(ans)[match.b:match.b + match.size]) / len(ans)


def levenshtein_distance(s1: str, s2: str) -> float:
    return nltk.edit_distance(s1, s2)


def norm_levenshtein_distance(s1: str, s2: str) -> float:
    return nltk.edit_distance(s1, s2) / len(s2)


def accuracy(resp: str, ans: str) -> float:
    if len(resp) == 0:
        return 0
    acc = 0
    for i in range(0, len(resp)):
        if i >= len(ans):
            continue
        elif resp[i] == ans[i]:
            acc = acc + 1
    return acc / len(resp)


def precision(resp: str, ans: str) -> float:
    if len(resp) == 0:
        return 0
    prec = sum(1 for c in resp if c in ans)
    return min(1.0, prec / len(resp))


def binary(resp: str or int, ans: str or int) -> bool:
    return resp == ans


def l1loss(resp: int, ans: int):
    return abs(resp - float(ans))


def rouge_l_funct(resp: str, ans: str) -> float:
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return scorer.score(ans, resp)['rougeL'].fmeasure


def cosine_similarity_funct(resp: str, ans: str) -> float:
    vectorizer = TfidfVectorizer().fit([resp, ans])
    vectors = vectorizer.transform([resp, ans])
    return cosine_similarity(vectors[0], vectors[1])[0, 0]


def bertscore_funct(resp: str, ans: str) -> float:
    bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    try:
        P, R, F1 = bert_scorer.score([resp], [ans])
        return F1[0].item()
    except Exception:
        return 0.0

def serialize_ground_truth_to_text(gt: Dict[str, Dict[str, Any]]) -> str:
    return json.dumps(gt, ensure_ascii=False, separators=(",", ":"))

def _best_pred_row(pred_obj: dict) -> dict:
    if not isinstance(pred_obj, dict) or not pred_obj:
        return {}
    if not all(isinstance(v, dict) for v in pred_obj.values()):
        return pred_obj
    def numeric_count(row):
        return sum(1 for k in TARGET_NUMERIC_FIELDS
                   if isinstance(row.get(k), (int, float)))
    return max(pred_obj.values(), key=numeric_count, default={})


def get_comp_bleu(candidate, reference):
    try:
        smoothie = SmoothingFunction().method4
        return sentence_bleu([reference.split()], candidate.split(),
                              smoothing_function=smoothie)
    except Exception:
        return 0.0


def _to_num(x) -> Optional[float]:
    try:
        return float(x) if isinstance(x, (int, float)) else float(str(x))
    except Exception:
        return None


def _coverage_by_gt(pred: dict, gt: dict) -> float:
    if not isinstance(gt, dict) or not gt:
        return 0.0
    gt_is_ts = all(isinstance(v, dict) for v in gt.values())
    if isinstance(pred, dict):
        pred = rekey_timeseries(pred, TARGET_NUMERIC_FIELDS)
    if gt_is_ts:
        gt = rekey_timeseries(gt, TARGET_NUMERIC_FIELDS)
    denom = (len(gt) * len(TARGET_NUMERIC_FIELDS)) if gt_is_ts else len(TARGET_NUMERIC_FIELDS)
    if denom == 0:
        return 0.0
    if not isinstance(pred, dict):
        return 0.0
    num = 0
    if gt_is_ts:
        for ts, _gt_row in gt.items():
            row = pred.get(ts, {}) if isinstance(pred.get(ts, {}), dict) else {}
            for k in TARGET_NUMERIC_FIELDS:
                if isinstance(row.get(k), (int, float)):
                    num += 1
    else:
        row = _best_pred_row(pred)
        for k in TARGET_NUMERIC_FIELDS:
            if isinstance(row.get(k), (int, float)):
                num += 1
    return (num / denom) * 100.0


def _l1loss_matched(pred: dict, gt: dict) -> Tuple[Optional[float], float, int]:
    """
    Matched MAE: computes MAE ONLY on timestamps that appear in BOTH
    the prediction and the ground truth. Missing timestamps are excluded
    from MAE and reflected in timestamp_coverage instead.

    Returns:
        mae               : float or None (None if no timestamps matched)
        timestamp_coverage: float 0-100 (% of GT timestamps matched)
        n_matched         : int (number of matched timestamps)

    This isolates pure forecasting quality from extraction failures.
    Use alongside _l1loss_aligned() to separate the two effects.
    """
    keys = TARGET_NUMERIC_FIELDS

    if not isinstance(gt, dict) or not gt:
        return None, 0.0, 0

    gt_is_ts = all(isinstance(v, dict) for v in gt.values())

    # Normalise timestamps
    if isinstance(pred, dict):
        pred = rekey_timeseries(pred, keys)
    else:
        pred = {}

    if gt_is_ts:
        gt = rekey_timeseries(gt, keys)

    if not gt_is_ts:
        # Single-row GT: check if pred has any matching row
        pred_row = _best_pred_row(pred)
        total, count = 0.0, 0
        for k in keys:
            gv = _to_num(gt.get(k))
            pv = _to_num(pred_row.get(k))
            if gv is None or pv is None:
                continue
            total += abs(pv - gv)
            count += 1
        mae = (total / count) if count else None
        ts_coverage = 100.0 if count > 0 else 0.0
        return mae, ts_coverage, (1 if count > 0 else 0)

    # Time-series GT: find intersection of timestamps
    gt_timestamps   = set(gt.keys())
    pred_timestamps = set(pred.keys()) if isinstance(pred, dict) else set()
    matched_ts      = gt_timestamps & pred_timestamps

    ts_coverage = (len(matched_ts) / len(gt_timestamps)) * 100.0 \
                  if gt_timestamps else 0.0

    if not matched_ts:
        return None, ts_coverage, 0

    total, count = 0.0, 0
    for ts in matched_ts:
        gt_row   = gt[ts]
        pred_row = pred.get(ts, {})
        pred_row = pred_row if isinstance(pred_row, dict) else {}
        for k in keys:
            gv = _to_num(gt_row.get(k))
            pv = _to_num(pred_row.get(k))
            if gv is None or pv is None:
                continue
            total += abs(pv - gv)
            count += 1

    mae = (total / count) if count else None
    return mae, ts_coverage, len(matched_ts)

def _l1loss_aligned(pred: dict, gt: dict) -> float:
    """
    Penalised MAE: missing or unmatched prediction timestamps are treated
    as predicting 0.0 for every field. This means extraction failures
    are penalised heavily, which conflates format failure with forecasting
    error. Used alongside matched MAE to expose this conflation.
    """
    keys = TARGET_NUMERIC_FIELDS
    if not isinstance(gt, dict) or not gt:
        return 0.0

    gt_is_ts = all(isinstance(v, dict) for v in gt.values())
    _rekey   = lambda x, *_: x if isinstance(x, dict) else {}

    pred = _rekey(pred, keys) if isinstance(pred, dict) else {}
    if gt_is_ts:
        gt = _rekey(gt, keys)

    def _best_row(p: dict) -> dict:
        best, best_c = {}, -1
        if isinstance(p, dict):
            for row in p.values():
                if not isinstance(row, dict):
                    continue
                c = sum(1 for k in keys if _to_num(row.get(k)) is not None)
                if c > best_c:
                    best, best_c = row, c
        return best

    total = 0.0
    count = 0

    if not gt_is_ts:
        prow = _best_row(pred)
        for k in keys:
            gv = _to_num(gt.get(k))
            if gv is None:
                continue
            pv = _to_num(prow.get(k))
            pv = pv if pv is not None else 0.0   # ← penalise missing
            total += abs(pv - gv)
            count += 1
        return (total / count) if count else 0.0

    for ts, gt_row in gt.items():
        prow = pred.get(ts, {})
        prow = prow if isinstance(prow, dict) else {}
        for k in keys:
            gv = _to_num(gt_row.get(k))
            if gv is None:
                continue
            pv = _to_num(prow.get(k))
            pv = pv if pv is not None else 0.0   # ← penalise missing
            total += abs(pv - gv)
            count += 1

    return (total / count) if count else 0.0

def score(resp, ans) -> Dict[str, float]:
    if type(resp) != type(ans):
        raise ValueError('Response and answer must be of the same type')
    if type(resp) == int:
        return {
            'levenshtein_distance': 0, 'norm_levenshtein_distance': 0,
            'lcs': 0, 'norm_lcs': 0, 'accuracy': 0, 'precision': 0,
            'binary': binary(resp, ans), 'l1loss': l1loss(resp, ans),
        }
    return {
        'levenshtein_distance':      levenshtein_distance(resp, ans),
        'norm_levenshtein_distance': norm_levenshtein_distance(resp, ans),
        'lcs':                       longest_common_substring(resp, ans),
        'norm_lcs':                  norm_longest_common_substring(resp, ans),
        'accuracy':                  accuracy(resp, ans),
        'precision':                 precision(resp, ans),
        'binary':                    binary(resp, ans),
        'l1loss':                    _l1loss_aligned(resp, ans),
        'bertscore':                 bertscore_funct(resp, ans),
        'cosinesim':                 cosine_similarity_funct(resp, ans),
        'bleu':                      get_comp_bleu(resp, ans),
        'rouge_l':                   rouge_l_funct(resp, ans),
    }


class Metrics:
    def __init__(self):
        self.__scores = {
            'levenshtein_distance':      0,
            'norm_levenshtein_distance': 0,
            'lcs':                       0,
            'norm_lcs':                  0,
            'accuracy':                  0,
            'precision':                 0,
            'binary':                    0,
            'l1loss':                    0,   # penalised MAE accumulator
            'l1loss_matched':            0,   # matched MAE accumulator
            'l1loss_matched_count':      0,   # how many samples had ≥1 matched ts
            'timestamp_coverage':        0,   # timestamp coverage accumulator
            'coverage':                  0,
            'bertscore':                 0,
            'cosinesim':                 0,
            'bleu':                      0,
            'rouge_l':                   0,
            'strict':                    0,
            'lenient':                   0,
            '__valid_json_count':        0,
        }
        self.__ext = {
            "n": 0,
            "tp": 0, "fp": 0, "fn": 0, "tn": 0,
            "type_correct":     0,
            "strict_correct":   0,
            "lenient_correct":  0,
            "consistent_correct": 0,
        }
        self.__n            = 0
        self.__last_subject = "weather"

    def update(self, resp: str, ans: str):
        self.__n += 1
        scores = score(resp, ans)
        for x in self.__scores:
            self.__scores[x] += scores.get(x, 0)
        return scores

    def update_weather(self, json_llm_response: dict, sentence_llm_response: str,
                       ground_truth: dict, result_sentence: str, mode_json: bool):
        """
        Evaluate one weather prediction.

        Computes and stores:
          l1loss            — penalised MAE (missing timestamps → 0.0)
          l1loss_matched    — matched MAE (only on aligned timestamps)
          timestamp_coverage— % of GT timestamps present in prediction
          coverage          — % of GT numeric fields populated
        """
        self.__n += 1
        scores   = {}

        # ── Text preparation ──────────────────────────────────────────────────
        target_text      = (json.dumps(ground_truth, ensure_ascii=False,
                                       separators=(",", ":"))
                            if mode_json else result_sentence)
        text_for_metrics = (json.dumps(json_llm_response, ensure_ascii=False,
                                       separators=(",", ":"))
                            if mode_json else sentence_llm_response)

        # ── Coverage (field-level) ────────────────────────────────────────────
        try:
            scores["coverage"] = _coverage_by_gt(json_llm_response, ground_truth)
        except Exception:
            scores["coverage"] = 0.0

        # ── Penalised MAE (original — missing ts penalised as 0.0) ───────────
        try:
            scores["l1loss"] = _l1loss_aligned(json_llm_response, ground_truth)
        except Exception:
            scores["l1loss"] = float("inf")

        # ── Matched MAE (NEW — only on aligned timestamps) ───────────────────
        try:
            mae_matched, ts_cov, n_matched = _l1loss_matched(json_llm_response, ground_truth)
            # Store None as 0.0 for accumulation; track count separately
            scores["l1loss_matched"]   = mae_matched if mae_matched is not None else 0.0
            scores["timestamp_coverage"] = ts_cov
            scores["n_matched"]          = n_matched   # 1 if any ts matched, else 0
        except Exception:
            scores["l1loss_matched"]     = 0.0
            scores["timestamp_coverage"] = 0.0
            scores["n_matched"]          = 0

        # ── Strict / Lenient ──────────────────────────────────────────────────
        penalty          = 0.5 + 0.5 * (scores["coverage"] / 100.0)
        scores["strict"] = 1.0
        scores["lenient"] = penalty * (
            1.0 / (1.0 + max(0.0, float(scores["l1loss"]))))

        # ── String metrics ────────────────────────────────────────────────────
        try:
            scores.update({
                "levenshtein_distance":      levenshtein_distance(text_for_metrics, target_text),
                "norm_levenshtein_distance": norm_levenshtein_distance(text_for_metrics, target_text),
                "lcs":                       longest_common_substring(text_for_metrics, target_text),
                "norm_lcs":                  norm_longest_common_substring(text_for_metrics, target_text),
                "bertscore":                 bertscore_funct(text_for_metrics, target_text),
                "cosinesim":                 cosine_similarity_funct(text_for_metrics, target_text),
                "bleu":                      get_comp_bleu(text_for_metrics, target_text),
                "rouge_l":                   rouge_l_funct(text_for_metrics, target_text),
            })
        except Exception:
            for k in ["levenshtein_distance", "norm_levenshtein_distance",
                      "lcs", "norm_lcs", "bertscore", "cosinesim", "bleu", "rouge_l"]:
                scores.setdefault(k, 0.0)

        # ── Accumulate ────────────────────────────────────────────────────────
        for k in self.__scores:
            if k == "l1loss_matched_count":
                # Count samples where at least one timestamp matched
                self.__scores[k] += 1 if scores.get("n_matched", 0) > 0 else 0
            elif k == "timestamp_coverage":
                self.__scores[k] += scores.get("timestamp_coverage", 0.0)
            else:
                self.__scores[k] += scores.get(k, 0)

        return scores

    def update_weather_extreme(self, pred_has: bool, pred_type: str,
                               gt_has: bool, gt_type: str):
        self.__last_subject = "weather_extreme"
        e = self.__ext
        e["n"] += 1

        if pred_has and gt_has:       e["tp"] += 1
        elif pred_has and not gt_has: e["fp"] += 1
        elif not pred_has and gt_has: e["fn"] += 1
        else:                         e["tn"] += 1

        lenient = (pred_has == gt_has)
        if lenient:
            e["lenient_correct"] += 1

        consistent = (
            (not pred_has and str(pred_type) == "NA") or
            (pred_has and str(pred_type) != "NA")
        )
        if consistent:
            e["consistent_correct"] += 1

        strict = (
            consistent and
            (pred_has == gt_has) and
            (not gt_has or str(pred_type) == str(gt_type))
        )
        if strict:
            e["strict_correct"] += 1

        if gt_has and pred_has and str(pred_type) == str(gt_type):
            e["type_correct"] += 1

        return {
            "strict":    1.0 if strict    else 0.0,
            "lenient":   1.0 if lenient   else 0.0,
            "consistent": 1.0 if consistent else 0.0,
        }

    def get(self, subject: str = None):
        subj = subject or self.__last_subject

        if subj == "weather-extreme":
            e  = self.__ext
            n  = e["n"]
            if n == 0:
                return {
                    "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                    "type_accuracy": 0.0, "strict": 0.0, "lenient": 0.0,
                    "consistency": 0.0, "tp": 0, "fp": 0, "fn": 0, "tn": 0,
                }
            tp, fp, fn, tn = e["tp"], e["fp"], e["fn"], e["tn"]
            acc  = (tp + tn) / n
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec  = tp / (tp + fn) if (tp + fn) else 0.0
            f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
            return {
                "accuracy":      acc,
                "precision":     prec,
                "recall":        rec,
                "f1":            f1,
                "type_accuracy": e["type_correct"] / tp if tp else 0.0,
                "strict":        e["strict_correct"]    / n,
                "lenient":       e["lenient_correct"]   / n,
                "consistency":   e["consistent_correct"] / n,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            }

        # ── Weather scores ────────────────────────────────────────────────────
        n = self.__n

        # Matched MAE: average only over samples that had ≥1 matched timestamp
        n_matched_samples = self.__scores["l1loss_matched_count"]
        mae_matched = (self.__scores["l1loss_matched"] / n_matched_samples
                       if n_matched_samples > 0 else None)

        return {
            "levenshtein_distance":      self.__scores["levenshtein_distance"]      / n,
            "norm_levenshtein_distance": self.__scores["norm_levenshtein_distance"] / n,
            "lcs":                       self.__scores["lcs"]                       / n,
            "norm_lcs":                  self.__scores["norm_lcs"]                  / n,
            # ── Two MAE variants ──────────────────────────────────────────────
            "l1loss":                    self.__scores["l1loss"]          / n,   # penalised
            "l1loss_matched":            mae_matched,                             # matched (None if 0 matched)
            "timestamp_coverage":        self.__scores["timestamp_coverage"] / n, # % ts matched
            # ── Other metrics ─────────────────────────────────────────────────
            "coverage":                  self.__scores["coverage"]         / n,
            "bertscore":                 self.__scores["bertscore"]        / n,
            "cosinesim":                 self.__scores["cosinesim"]        / n,
            "bleu":                      self.__scores["bleu"]             / n,
            "rouge_l":                   self.__scores["rouge_l"]          / n,
            "strict":                    self.__scores["strict"]           / n,
            "lenient":                   self.__scores["lenient"]          / n,
        }

    # ── Individual getters (kept for backward compatibility) ─────────────────
    def get_l1loss(self):             return self.__scores["l1loss"]             / self.__n
    def get_l1loss_matched(self):
        n = self.__scores["l1loss_matched_count"]
        return self.__scores["l1loss_matched"] / n if n > 0 else None
    def get_timestamp_coverage(self): return self.__scores["timestamp_coverage"] / self.__n
    def get_levenshtein(self):        return self.__scores["levenshtein_distance"]      / self.__n
    def get_norm_levenshtein(self):   return self.__scores["norm_levenshtein_distance"] / self.__n
    def get_lcs(self):                return self.__scores["lcs"]       / self.__n
    def get_norm_lcs(self):           return self.__scores["norm_lcs"]  / self.__n
    def get_accuracy(self):           return self.__scores["accuracy"]  / self.__n
    def get_precision(self):          return self.__scores["precision"] / self.__n
    def get_binary(self):             return self.__scores["binary"]    / self.__n
    def get_bertscore(self):          return self.__scores["bertscore"] / self.__n
    def get_cosinesim(self):          return self.__scores["cosinesim"] / self.__n
    def get_rouge_l(self):            return self.__scores["rouge_l"]   / self.__n
    def get_bleu(self):               return self.__scores["bleu"]      / self.__n
    def get_coverage(self):           return self.__scores["coverage"]  / self.__n
    def get_strict_score(self):       return self.__scores["strict"]    / self.__n
    def get_strict_lenient(self):     return self.__scores["lenient"]   / self.__n


if __name__ == '__main__':
    result = """Based on the given weather observations, I predict that in the next 2 hours (from 09:00 to 11:00), the weather will be as follows:

{
  "2014-03-03 10:00:00": {
    "east_west_wind_speed_10m": 1.25,
    "north_south_wind_speed_10m": -2.89,
    "dewpoint_temperature_2m": 21.69,
    "air_temperature_2m": 27.03,
    "mean_sea_level_pressure": 1009.33,
    "surface_pressure": 987.92,
    "total_precipitation": 0.0,
    "latitude": null,
    "longitude": null,
    "area": "Pulau Morotai, North Maluku"
  },
  "2014-03-03 11:00:00": {
    "east_west_wind_speed_10m": 1.13,
    "north_south_wind_speed_10m": -2.93,
    "dewpoint_temperature_2m": 21.51,
    "air_temperature_2m": 26.65,
    "mean_sea_level_pressure": 1009.19,
    "surface_pressure": 987.43,
    "total_precipitation": 0.0,
    "latitude": null,
    "longitude": null,
    "area": "Pulau Morotai, North Maluku"
  }
}

Based on the trends and patterns observed in the given weather observations, I predict that the wind speed will continue to decrease, and the air temperature will drop slightly. The mean sea level pressure and surface pressure are expected to remain relatively stable.

    """
    ground_truth = """{'2014-03-03 10:00:00': {'east_west_wind_speed_10m': 1.25, 'north_south_wind_speed_10m': -2.67, 'dewpoint_temperature_2m': 22.15, 'air_temperature_2m': 26.29, 'mean_sea_level_pressure': 1010.29, 'surface_pressure': 988.94, 'total_precipitation': 0.0, 'latitude': 2.25, 'longitude': 128.5, 'area': 'Pulau Morotai, North Maluku'}, '2014-03-03 11:00:00': {'east_west_wind_speed_10m': 1.02, 'north_south_wind_speed_10m': -2.65, 'dewpoint_temperature_2m': 22.26, 'air_temperature_2m': 25.75, 'mean_sea_level_pressure': 1010.93, 'surface_pressure': 989.54, 'total_precipitation': 0.0, 'latitude': 2.25, 'longitude': 128.5, 'area': 'Pulau Morotai, North Maluku'}}"""

    gt_obj = ast.literal_eval(ground_truth)
    # llm_result = clean_json_template(result)

    print("type ", type(gt_obj))
    metrics = Metrics()

    # m = Metrics()
    #
    # # Case A: No + Funnel Cloud (INKONSISTEN)
    # print("\nCase A: No + Funnel Cloud (should strict=0, lenient=1)")
    # out = m.update_weather_extreme(
    #     pred_has=False,
    #     pred_type="Funnel Cloud",
    #     gt_has=False,
    #     gt_type="NA"
    # )
    # print(out)
    #
    # # Reset metrics
    # m = Metrics()
    #
    # # Case B: Yes + Flood (BENAR)
    # print("\nCase B: Yes + Flood (should strict=1, lenient=1)")
    # out = m.update_weather_extreme(
    #     pred_has=True,
    #     pred_type="Flood",
    #     gt_has=True,
    #     gt_type="Flood"
    # )
    # print(out)
    #
    # # Reset metrics
    # m = Metrics()
    #
    # # Case C: Yes + Wrong Type
    # print("\nCase C: Yes + Wrong Type (should strict=0, lenient=1)")
    # out = m.update_weather_extreme(
    #     pred_has=True,
    #     pred_type="Tornado",
    #     gt_has=True,
    #     gt_type="Flood"
    # )
    # print(out)
    # async def _run():
    #     out = await metrics.update_weather(
    #         prediction_text=result,
    #         ground_truth=gt_obj,
    #         reference_text=None
    #     )
    #     return out

    # scores = asyncio.run(_run())
    # print(json.dumps(scores, ensure_ascii=False, indent=2))
    #
    # print("\n=== Individual Example Evaluation (update_weather) ===")
    # for k, v in scores.items():
    #     print(f"{k:30s}: {v:.4f}" if isinstance(v, float) else f"{k:30s}: {v}")
    #
    # print("\n=== Aggregated Average After 1 Example ===")
    # for k, v in metrics.get().items():
    #     print(f"{k:30s}: {v:.4f}")

    # test_ans = 'The quick brown fox jumps over the lazy dog'
    # test_correct_resp = test_ans
    # test_wrong_resp = 'The quick brown fox jumps over the lazy cat'
    # test_wrong_resp2 = 'The quick brown fox jumps over the lazy dog and the cat'
    # test_empty_resp = ''
    # assert score(test_correct_resp, test_ans)['binary'] == True
    # assert score(test_wrong_resp, test_ans)['binary'] == False
    # assert score(test_correct_resp, test_ans)['levenshtein_distance'] == 0
    # assert score(test_wrong_resp, test_ans)['levenshtein_distance'] == 3 / len(test_ans)
    # assert score(test_correct_resp, test_ans)['lcs'] == 1
    # assert score(test_wrong_resp, test_ans)['lcs'] == len(test_ans[:-3])/len(test_ans)
    # assert score(test_correct_resp, test_ans)['accuracy'] == 1
    # assert score(test_wrong_resp, test_ans)['accuracy'] == len(test_ans[:-3])/len(test_ans)
    # assert score(test_wrong_resp2, test_ans)['accuracy'] == len(test_ans)/len(test_wrong_resp2)
    # assert score(test_correct_resp, test_ans)['precision'] == 1
    # assert score(test_wrong_resp, test_ans)['precision'] == 1
    #
    # metrics = Metrics()
    # metrics.update(test_correct_resp, test_ans)
    # metrics.update(test_empty_resp, test_ans)
    # assert metrics.get()['binary'] == 0.5
    # assert metrics.get()['levenshtein_distance'] == 0.5
    # assert metrics.get()['lcs'] == 0.5
    # assert metrics.get()['accuracy'] == 0.5
    # assert metrics.get()['precision'] == 0.5
    #
    # test_ans = 'The quick brown fox jumps over the lazy dog'
    # test_responses = {
    #     'Correct': test_ans,
    #     'Wrong (cat)': 'The quick brown fox jumps over the lazy cat',
    #     'Extended': 'The quick brown fox jumps over the lazy dog and the cat',
    #     'Empty': ''
    # }
    #
    # print("=== Testing `score()` function ===")
    # for name, resp in test_responses.items():
    #     print("name ", name)
    #     print("resp ", resp)
    #     print(f"\n--- Response: {name} ---")
    #     scores = score(resp, test_ans)
    #     for metric, value in scores.items():
    #         if isinstance(value, float):
    #             print(f"{metric:30s}: {value:.4f}")
    #         else:
    #             print(f"{metric:30s}: {value}")
    #
    # print("\n=== Testing `Metrics` class aggregation ===")
    # metrics = Metrics()
    # for name, resp in test_responses.items():
    #     metrics.update(resp, test_ans)
    #
    # aggregated = metrics.get()
    # print("\n--- Aggregated Metrics ---")
    # for metric, value in aggregated.items():
    #     print(f"{metric:30s}: {value:.4f}")

