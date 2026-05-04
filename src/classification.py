import csv
import os
from dataclasses import dataclass
from typing import Dict, List


def count_coins(circles):
    return int(len(circles))


def load_ground_truth_counts(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Ground truth file not found: {csv_path}")

    counts = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"image_name", "coins_count"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("Ground truth CSV must include: image_name,coins_count")
        for row in reader:
            image_name = (row.get("image_name") or "").strip()
            count = int(row.get("coins_count") or 0)
            if image_name:
                counts[image_name] = count
    return counts


def evaluate_count_precision_recall(pred_count, gt_count):
    if gt_count is None:
        return None

    tp = min(pred_count, gt_count)
    fp = max(0, pred_count - gt_count)
    fn = max(0, gt_count - pred_count)
    precision = tp / pred_count if pred_count > 0 else 0.0
    recall = tp / gt_count if gt_count > 0 else 0.0

    return {
        "pred_count": pred_count,
        "gt_count": gt_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "exact_match": pred_count == gt_count,
        "abs_error": abs(pred_count - gt_count),
    }


@dataclass
class CoinRule:
    min_r_norm: float
    max_r_norm: float
    min_hue: float
    max_hue: float


class DenominationClassifier:
    # Radius ratio + HSV rules only
    RULES: Dict[str, CoinRule] = {
        "QUARTER": CoinRule(0.110, 0.220, 0, 45),
        "DIME": CoinRule(0.085, 0.130, 0, 45),
        "PENNY": CoinRule(0.050, 0.095, 70, 160),
    }

    def _rule_score(self, denom: str, r_norm: float, mean_hue: float) -> float:
        rule = self.RULES[denom]

        if rule.min_r_norm <= r_norm <= rule.max_r_norm:
            r_score = 1.0
        else:
            r_center = (rule.min_r_norm + rule.max_r_norm) / 2.0
            r_span = max(1e-6, (rule.max_r_norm - rule.min_r_norm) / 2.0)
            r_score = max(0.0, 1.0 - abs(r_norm - r_center) / (2.5 * r_span))

        if rule.min_hue <= mean_hue <= rule.max_hue:
            h_score = 1.0
        else:
            h_center = (rule.min_hue + rule.max_hue) / 2.0
            h_span = max(1e-6, (rule.max_hue - rule.min_hue) / 2.0)
            h_score = max(0.0, 1.0 - abs(mean_hue - h_center) / (2.5 * h_span))

        return 0.6 * r_score + 0.4 * h_score

    def classify_coin(self, feature: dict):
        r_norm = float(feature["r_norm"])
        mean_hue = float(feature["mean_hue"])

        best_label = "UNKNOWN"
        best_score = -1.0

        for denom in self.RULES.keys():
            score = self._rule_score(denom, r_norm, mean_hue)
            if score > best_score:
                best_score = score
                best_label = denom

        if best_score < 0.40:
            best_label = "UNKNOWN"

        out = dict(feature)
        out["denomination"] = best_label
        out["denom_conf"] = float(max(0.0, best_score))
        return out

    def classify_all(self, features: List[dict]) -> List[dict]:
        return [self.classify_coin(feature) for feature in features]
