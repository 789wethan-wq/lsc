"""M2 acceptance: parameter recovery on well-specified DGPs.

For each model/DGP pair, fit on n_train obs over n_reps seeded
replications and check that the mean estimate is within 2 Monte Carlo
standard errors of the truth. Writes a LaTeX + CSV table to
paper_assets/.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from lsc.dgp import LocalLevelDGP, MarkovSwitchingDGP
from lsc.models import HMMModel, KalmanModel


def recover_kalman(n_reps: int, T: int, q: float, r: float, seed0: int) -> list[dict]:
    est_q, est_r = [], []
    dgp = LocalLevelDGP(q=q, r=r)
    for i in range(n_reps):
        s = dgp.sample(T, seed=seed0 + i)
        m = KalmanModel().fit(s.Y)
        est_r.append(m._params[0])  # sigma2.irregular
        est_q.append(m._params[1])  # sigma2.level
    rows = []
    for name, truth, est in [("sigma2.level (q)", q, est_q),
                             ("sigma2.irregular (r)", r, est_r)]:
        est = np.asarray(est)
        mc_se = est.std(ddof=1) / np.sqrt(n_reps)
        rows.append(dict(model="kalman_llevel", dgp=f"local_level(q={q},r={r})",
                         param=name, truth=truth, mean_est=est.mean(),
                         mc_se=mc_se, within_2se=abs(est.mean() - truth) <= 2 * mc_se))
    return rows


def recover_hmm(n_reps: int, T: int, seed0: int) -> list[dict]:
    means, sigmas, p = (0.0, 3.0), (1.0, 1.0), 0.97
    dgp = MarkovSwitchingDGP(means=means, sigmas=sigmas, persistence=p)
    acc = {k: [] for k in ("mu0", "mu1", "sig0", "sig1", "p00")}
    for i in range(n_reps):
        s = dgp.sample(T, seed=seed0 + i)
        m = HMMModel(2, em_seed=i).fit(s.Y)
        acc["mu0"].append(m.mu[0]); acc["mu1"].append(m.mu[1])
        acc["sig0"].append(np.sqrt(m.sig2[0])); acc["sig1"].append(np.sqrt(m.sig2[1]))
        acc["p00"].append(m.A[0, 0])
    truths = dict(mu0=means[0], mu1=means[1], sig0=sigmas[0], sig1=sigmas[1], p00=p)
    rows = []
    for k, vals in acc.items():
        vals = np.asarray(vals)
        mc_se = vals.std(ddof=1) / np.sqrt(n_reps)
        rows.append(dict(model="hmm_2", dgp="markov_switching(0/3,p=.97)",
                         param=k, truth=truths[k], mean_est=vals.mean(),
                         mc_se=mc_se, within_2se=abs(vals.mean() - truths[k]) <= 2 * mc_se))
    return rows


def main(n_reps: int = 200) -> pd.DataFrame:
    rows = []
    rows += recover_kalman(n_reps, T=2000, q=0.5, r=1.0, seed0=10_000)
    rows += recover_kalman(n_reps, T=2000, q=0.1, r=1.0, seed0=20_000)
    rows += recover_hmm(n_reps, T=2000, seed0=30_000)
    df = pd.DataFrame(rows)
    df.to_csv("paper_assets/m2_param_recovery.csv", index=False)
    with open("paper_assets/m2_param_recovery.tex", "w") as f:
        f.write(df.to_latex(index=False, float_format="%.4f"))
    return df


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    df = main(n)
    print(df.to_string(index=False))
    ok = df["within_2se"].all()
    print("\nALL WITHIN 2 MC-SE:" , ok)
    sys.exit(0 if ok else 1)
