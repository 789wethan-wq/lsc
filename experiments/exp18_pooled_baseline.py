"""exp18_pooled_baseline.py -- "always-ARIMA" / "always-raw" fixed-rule
baselines for exp14's mixed-channel test.

exp14 (`experiments/exp14_mixed_channel.py`) reports, per SNR, the
combined statistic's detection rate against whichever single detector
is best AT THAT SNR -- an oracle comparison a real practitioner facing
unknown SNR cannot make after the fact. The missing number is what a
FIXED rule ("always run raw", "always run ARIMA", regardless of SNR)
would actually score, pooled across the SNR range exp14 already
covers.

No new simulation: exp14 already reports per-SNR detection rates for
raw, ARIMA, and the combined statistic at SNR in {0.1, 0.5, 2.0}
(`paper_assets/exp14_mixed_channel.csv`, n_eval=300 per SNR). This
script only pools those three already-computed numbers.

The pooling weight is a real methodological choice, not a formality:
equal-thirds across the three tested SNRs is the simplest defensible
default (no claim about the true population mix of SNRs in practice),
stated explicitly rather than presented as an intrinsic ground truth.

Reports four numbers side by side: always-raw, always-ARIMA,
combined-statistic (already jointly-calibrated in exp14), and
oracle-best-per-SNR (pick whichever of raw/ARIMA is better AT EACH
SNR, then pool) -- so the reader sees exactly what is gained and lost
at each level of information the practitioner is assumed to have.
Consistency check: oracle-best-per-SNR must be >= both fixed rules at
EVERY SNR by construction (it's a per-SNR max), not just on average.

Usage: python experiments/exp18_pooled_baseline.py
Output: prints the four pooled numbers with SEs; writes
paper_assets/exp18_pooled_baseline.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
IN_PATH = REPO_ROOT / "paper_assets" / "exp14_mixed_channel.csv"
OUT_PATH = REPO_ROOT / "paper_assets" / "exp18_pooled_baseline.csv"


def se_of_rate(p: float, n: int) -> float:
    """Binomial SE of a single detection-rate estimate."""
    return float(np.sqrt(p * (1 - p) / n))


def main() -> None:
    df = pd.read_csv(IN_PATH).sort_values("snr").reset_index(drop=True)
    assert list(df.snr) == [0.1, 0.5, 2.0], f"unexpected SNR grid: {list(df.snr)}"
    n_eval = int(df.n_eval.iloc[0])
    assert (df.n_eval == n_eval).all(), "pooling assumes equal n_eval per SNR"

    weights = np.full(len(df), 1.0 / len(df))  # equal-thirds, explicit assumption

    def pooled(col: str) -> tuple[float, float]:
        p_per_snr = df[col].to_numpy()
        pooled_rate = float((weights * p_per_snr).sum())
        # weights are fixed constants (not estimated), so the pooled SE is
        # just the weighted RMS of the per-SNR independent-sample SEs —
        # the three exp14 SNR cells use disjoint eval-seed blocks per SNR
        # (same seed base 500_000+i reused across SNR, but distinct DGPs),
        # so the three rate estimates are independent draws.
        se_per_snr = np.array([se_of_rate(p, n_eval) for p in p_per_snr])
        pooled_se = float(np.sqrt((weights ** 2 * se_per_snr ** 2).sum()))
        return pooled_rate, pooled_se

    raw_rate, raw_se = pooled("raw")
    arima_rate, arima_se = pooled("arima")
    combined_rate, combined_se = pooled("combined")

    oracle_per_snr = df[["raw", "arima"]].max(axis=1).to_numpy()
    oracle_rate = float((weights * oracle_per_snr).sum())
    oracle_se_per_snr = np.array([se_of_rate(p, n_eval) for p in oracle_per_snr])
    oracle_se = float(np.sqrt((weights ** 2 * oracle_se_per_snr ** 2).sum()))

    # consistency check: oracle-best-per-SNR >= both fixed rules AT EVERY SNR
    ok_vs_raw = bool((oracle_per_snr >= df["raw"].to_numpy() - 1e-12).all())
    ok_vs_arima = bool((oracle_per_snr >= df["arima"].to_numpy() - 1e-12).all())

    out = pd.DataFrame([
        dict(rule="always_raw", pooled_rate=raw_rate, pooled_se=raw_se),
        dict(rule="always_arima", pooled_rate=arima_rate, pooled_se=arima_se),
        dict(rule="combined_statistic", pooled_rate=combined_rate, pooled_se=combined_se),
        dict(rule="oracle_best_per_snr", pooled_rate=oracle_rate, pooled_se=oracle_se),
    ])
    out.attrs["weighting"] = "equal-thirds across SNR in {0.1, 0.5, 2.0}"
    out.to_csv(OUT_PATH, index=False)

    print("Pooling weight: equal-thirds across SNR in {0.1, 0.5, 2.0} "
          "(explicit assumption, not derived).")
    print(f"n_eval per SNR: {n_eval}\n")
    print("per-SNR detection rates feeding the pool:")
    print(df[["snr", "raw", "arima", "combined"]].to_string(index=False))
    print("\npooled (equal-thirds) rates:")
    print(out.round(4).to_string(index=False))
    print(f"\nconsistency check (oracle >= raw at every SNR): {ok_vs_raw}")
    print(f"consistency check (oracle >= arima at every SNR): {ok_vs_arima}")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
