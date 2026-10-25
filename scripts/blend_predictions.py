"""Blend the three model outputs using the original notebook's rank weights."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_predictions(path):
    """Load predictions and reject missing IDs, duplicate IDs, or invalid scores."""
    frame = pd.read_csv(path)
    required = ["row_id", "rule_violation"]
    if not set(required).issubset(frame.columns):
        raise ValueError(f"{path}: expected columns {required}")
    frame = frame[required].copy()
    if frame.empty:
        raise ValueError(f"{path}: predictions must not be empty")
    if frame["row_id"].isna().any() or frame["row_id"].duplicated().any():
        raise ValueError(f"{path}: row_id must be non-null and unique")
    frame["rule_violation"] = pd.to_numeric(frame["rule_violation"], errors="raise")
    if not np.isfinite(frame["rule_violation"].to_numpy(dtype=float)).all():
        raise ValueError(f"{path}: scores must be finite")
    return frame.sort_values("row_id").reset_index(drop=True)


def blend_predictions(deberta_path, gte_path, llm_path):
    """Return the original 0.5/0.3/0.7 weighted rank sum, aligned by row_id."""
    deberta, gte, llm = [
        read_predictions(path) for path in (deberta_path, gte_path, llm_path)
    ]
    expected_ids = deberta["row_id"]
    for name, frame in (("GTE", gte), ("LLM", llm)):
        if not expected_ids.equals(frame["row_id"]):
            raise ValueError(f"{name}: row_id values do not match the DeBERTa output")
    denominator = len(deberta) + 1
    scores = sum(
        weight * frame["rule_violation"].rank(method="average") / denominator
        for weight, frame in ((0.5, deberta), (0.3, gte), (0.7, llm))
    )
    return pd.DataFrame({"row_id": expected_ids, "rule_violation": scores})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deberta", type=Path, default=Path("sub_deberta_v3.csv"))
    parser.add_argument("--gte", type=Path, default=Path("sub_gte.csv"))
    parser.add_argument("--llm", type=Path, default=Path("sub_llm.csv"))
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()
    result = blend_predictions(args.deberta, args.gte, args.llm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} predictions to {args.output}")


if __name__ == "__main__":
    main()
