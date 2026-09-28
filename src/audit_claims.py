"""Final audit: every manuscript number cross-checked against raw outputs,
plus two reviewer-defence robustness computations.

Outputs a PASS/FAIL table. Exit 1 if any claim fails.
"""
import json
import re
import sys

import numpy as np

from config import DATA_DIR, RESULTS_DIR, ROOT

INT = RESULTS_DIR / "interventions"
failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        failures.append(name)


# --- A. margin map facts ---------------------------------------------------
res = {}
for p in sorted(RESULTS_DIR.glob("[0-9]*.json")):
    r = json.loads(p.read_text())
    if "nn_minus_gbdt" in r:
        res[r["dataset_id"]] = r
margins = {d: r["nn_minus_gbdt"] for d, r in res.items()}
ties = [d for d, m in margins.items() if abs(m) < 0.005]
decided = [d for d, m in margins.items() if abs(m) >= 0.005]
nn_wins = [d for d in decided if margins[d] > 0]
mean_dec = float(np.mean([margins[d] for d in decided]))
check("n_datasets=61", len(res) == 61, f"got {len(res)}")
check("ties=26/61", len(ties) == 26, f"got {len(ties)}")
check("decided=35, nn=9, gbdt=26", len(decided) == 35 and len(nn_wins) == 9,
      f"decided={len(decided)} nn={len(nn_wins)}")
check("mean decided margin ~ -0.028", abs(mean_dec - (-0.0281)) < 0.0015,
      f"actual {mean_dec:+.4f}")
mad = [r for r in res.values() if r["name"] == "madelon"][0]["nn_minus_gbdt"]
check("madelon ~ -0.32", abs(mad - (-0.322)) < 0.001, f"actual {mad:+.4f}")

sizes = {}
for d in res:
    meta = json.loads((DATA_DIR / str(d) / "meta.json").read_text())
    sizes[d] = meta["n_rows"]
top5 = sorted(sizes, key=sizes.get, reverse=True)[:5]
flagged = {151, 1461, 1590}
check("flagged canary datasets are 3 of the 5 largest",
      flagged <= set(top5),
      f"top5 by rows: {[(d, sizes[d]) for d in top5]}")

# --- B. descriptor count ---------------------------------------------------
dfile = next((RESULTS_DIR / "descriptors").glob("*.json"))
ndesc = len([k for k in json.loads(dfile.read_text())
             if k not in ("dataset_id", "name")])
check("manuscript descriptor count matches", ndesc == 22,
      f"actual {ndesc} (manuscript says 22)")

# --- C. report-vs-manuscript numeric claims --------------------------------
h1b = json.loads((RESULTS_DIR / "h1b_report.json").read_text())
h1c = json.loads((RESULTS_DIR / "h1c_report.json").read_text())
h6 = json.loads((RESULTS_DIR / "h6_report.json").read_text())
h7 = json.loads((RESULTS_DIR / "h7_report.json").read_text())
h8 = json.loads((RESULTS_DIR / "h8_report.json").read_text())
h5 = json.loads((RESULTS_DIR / "h5_report.json").read_text())
h9 = json.loads((RESULTS_DIR / "h9_report.json").read_text())
h2s = json.loads((RESULTS_DIR / "h2s_report.json").read_text())

claims = [
    ("H1b acc 0.686", h1b["lodo_accuracy_decided"], 0.686, 0.001),
    ("H1b baseline 0.743", h1b["majority_baseline_decided"], 0.743, 0.001),
    ("H1b p 0.318", h1b["perm_pvalue"], 0.318, 0.001),
    ("H1c rho 0.315", h1c["lodo_spearman_rho"], 0.315, 0.001),
    ("H1c p 0.025", h1c["perm_pvalue"], 0.025, 0.001),
    ("noise pilot delta -0.068",
     h6["pilot"]["noise_features"]["mean_delta_at_max"], -0.068, 0.001),
    ("noise ext delta -0.069",
     h6["extension"]["noise_features"]["mean_delta_at_max"], -0.069, 0.001),
    ("noise pilot holm 7.3e-4",
     h6["pilot"]["noise_features"]["holm_p"], 0.00073, 0.0001),
    ("noise ext holm 7.3e-4",
     h6["extension"]["noise_features"]["holm_p"], 0.00073, 0.0001),
    ("rotate pooled +0.011",
     h6["pooled"]["rotate"]["mean_delta_at_max"], 0.011, 0.001),
    ("rotate pooled holm 0.049",
     h6["pooled"]["rotate"]["holm_p"], 0.049, 0.0005),
    ("quantize pooled +0.002",
     h6["pooled"]["quantize"]["mean_delta_at_max"], 0.002, 0.001),
    ("H7 delta 0.024", h7["primary_mean_delta_at_0.25"], 0.024, 0.001),
    ("H7 p 0.039", h7["primary_wilcoxon_p_onesided"], 0.039, 0.001),
    ("H7 madelon +0.103", h7["madelon_spotlight"]["delta_at_0.25"],
     0.103, 0.001),
    ("H7 madelon +0.144", h7["madelon_spotlight"]["delta_at_0.1"],
     0.144, 0.001),
    ("H8 p 0.56", h8["primary_wilcoxon_p_onesided"], 0.5613, 0.005),
    ("H5 rho 0.47", h5["pooled"]["spearman_rho"], 0.473, 0.002),
    ("H9 rho 0.51",
     h9["primary_extension_replication"]["spearman_rho"], 0.506, 0.002),
    ("H9 select rho -0.14",
     h9["exploratory_select_features"]["spearman_rho"], -0.139, 0.002),
    ("H2s mean -0.110", h2s["mean_delta_retuned"], -0.110, 0.001),
]
for name, actual, expected, tol in claims:
    check(name, abs(actual - expected) <= tol, f"actual {actual:.5f}")

pooled_noise_p = h6["pooled"]["noise_features"]["wilcoxon_delta_p"]
check("pooled noise p < 1e-4", pooled_noise_p < 1e-4,
      f"actual {pooled_noise_p:.2e}")

# --- D. reviewer defence: family-wise decomposition ------------------------
noise_rows, rescue_rows = [], []
for f in INT.glob("*_noise_features_4.0.json"):
    c = json.loads(f.read_text())
    b = res[c["dataset_id"]]["families"]
    mlp0 = b["mlp"]["test_auc_mean"]
    gb0 = max(b["xgb"]["test_auc_mean"], b["lgbm"]["test_auc_mean"])
    noise_rows.append((c["families"]["mlp"]["test_auc_mean"] - mlp0,
                       max(c["families"]["xgb"]["test_auc_mean"],
                           c["families"]["lgbm"]["test_auc_mean"]) - gb0))
for f in INT.glob("*_select_features_0.25.json"):
    c = json.loads(f.read_text())
    b = res[c["dataset_id"]]["families"]
    mlp0 = b["mlp"]["test_auc_mean"]
    gb0 = max(b["xgb"]["test_auc_mean"], b["lgbm"]["test_auc_mean"])
    rescue_rows.append((c["families"]["mlp"]["test_auc_mean"] - mlp0,
                        max(c["families"]["xgb"]["test_auc_mean"],
                            c["families"]["lgbm"]["test_auc_mean"]) - gb0))
nm, ng = (np.mean([r[0] for r in noise_rows]),
          np.mean([r[1] for r in noise_rows]))
rm, rg = (np.mean([r[0] for r in rescue_rows]),
          np.mean([r[1] for r in rescue_rows]))
print(f"[INFO] noise@4x decomposition: MLP {nm:+.4f}, best-GBDT {ng:+.4f} "
      f"(n={len(noise_rows)})")
print(f"[INFO] rescue@0.25 decomposition: MLP {rm:+.4f}, best-GBDT {rg:+.4f} "
      f"(n={len(rescue_rows)})")
check("noise margin driven by MLP drop (not GBDT gain)", nm < ng <= 0.005,
      "")

# --- E. reviewer defence: single-descriptor LODO baseline ------------------
from meta_learner import build_meta_table
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import spearmanr
df = build_meta_table()
feats = [c for c in df.columns
         if c not in ("dataset_id", "name", "nn_minus_gbdt", "nn_wins")]
med = df[feats].median(numeric_only=True)
y = df["nn_minus_gbdt"].values
singles = {}
for f in feats:
    X1 = df[[f]].fillna(med[f]).values
    preds = np.zeros(len(y))
    for i in range(len(y)):
        mask = np.arange(len(y)) != i
        reg = GradientBoostingRegressor(random_state=0, max_depth=2,
                                        n_estimators=150)
        reg.fit(X1[mask], y[mask])
        preds[i] = reg.predict(X1[i:i + 1])[0]
    singles[f] = float(spearmanr(preds, y).statistic)
best = sorted(singles.items(), key=lambda kv: -kv[1])[:3]
print(f"[INFO] top single-descriptor LODO rho: {best}")
tex_now = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
check("single-descriptor superiority (rho=0.49) disclosed in manuscript",
      "0.49" in tex_now and "overfits" in tex_now,
      f"best single {best[0]}")

# --- G. revision re-analyses (h10, h12) quoted in the revised text ---------
h10 = json.loads((RESULTS_DIR / "h10_report.json").read_text())
h12 = json.loads((RESULTS_DIR / "h12_report.json").read_text())
ta, da = h12["A_tie_sensitivity"], h12["C_decision_analysis"]
al = h12["B_alignment_clustered"]
reg = da["mean_regret_auc"]
rev_claims = [
    ("realistic noise -0.072", h10["mean_delta_realistic"], -0.072, 0.001),
    ("realistic vs gaussian -0.003",
     h10["paired_realistic_minus_gaussian_mean"], -0.003, 0.001),
    ("realistic paired p 0.41", h10["paired_wilcoxon_p_twosided"], 0.41,
     0.005),
    ("seed SE median 0.0009", ta["seed_se_median"], 0.0009, 0.0001),
    ("ties also within 2SE = 5", ta["ties_005_also_2se_tie"], 5, 0),
    ("ties at AUC>0.99 = 21", ta["ties_005_below_1pct_headroom"], 21, 0),
    ("relative 5% ties = 8",
     [r for r in ta["rules"] if r["rule"] == "|d|<0.05*(1-AUC)"][0]
     ["n_ties"], 8, 0),
    ("acc/AUC spearman 0.83",
     ta["other_metrics"]["spearman_auc_vs_acc_margin"], 0.83, 0.005),
    ("ll/AUC spearman 0.77",
     ta["other_metrics"]["spearman_auc_vs_logloss_margin"], 0.77, 0.005),
    ("regret always GBDT 0.0024", reg["always_gbdt"], 0.0024, 0.00005),
    ("regret always MLP 0.0185", reg["always_mlp"], 0.0185, 0.00005),
    ("regret meta 0.0026", reg["meta_regressor_sign"], 0.0026, 0.00005),
    ("regret T_mi 0.0027", reg["T_mi_max_sign"], 0.0027, 0.00005),
    ("top-25% capture 26%", [c for c in da["prioritisation_curve"]
                             if c["frac_datasets_mlp_trained"] == 0.25][0]
     ["gain_captured_meta"], 0.26, 0.005),
    ("top-25% p 0.44", da["top25pct_perm_p_meta"], 0.44, 0.005),
    ("pilot align rho 0.47", al["pilot"]["all"]["rho"], 0.47, 0.005),
    ("extension align rho 0.51", al["extension"]["all"]["rho"], 0.51, 0.005),
]
for name, actual, expected, tol in rev_claims:
    check(name, abs(actual - expected) <= tol, f"actual {actual:.5f}")
check("acc sign agreement 32/35",
      ta["other_metrics"]["sign_agree_acc_on_auc_decided"] == "32/35", "")
check("ll sign agreement 31/35",
      ta["other_metrics"]["sign_agree_ll_on_auc_decided"] == "31/35", "")
check("extension without-noise CI covers 0",
      al["extension"]["without_noise"]["cluster_bootstrap_ci95"][0] < 0, "")
check("dataset-exchange test significant only when pooled",
      al["pooled"]["all"]["dataset_identity_perm_p"] < 0.05
      and al["pilot"]["all"]["dataset_identity_perm_p"] >= 0.05
      and al["extension"]["all"]["dataset_identity_perm_p"] >= 0.05, "")
# --- H. confirmatory protocol B (h13) and protocol-A architectures (h11) ---
h13 = json.loads((RESULTS_DIR / "h13_report.json").read_text())
h11 = json.loads((RESULTS_DIR / "h11_report.json").read_text())["archs"]
b_claims = [
    ("P1 mlp -0.032", h13["P1"]["mlp"]["mean"], -0.032, 0.0005),
    ("P1 resnet -0.034", h13["P1"]["resnet"]["mean"], -0.034, 0.0005),
    ("P1 new mlp -0.016", h13["P1_new"]["mlp"]["mean"], -0.016, 0.0005),
    ("P1 new resnet -0.018", h13["P1_new"]["resnet"]["mean"], -0.018,
     0.0005),
    ("P4 mlp +0.019", h13["P4"]["mlp"]["mean"], 0.019, 0.0005),
    ("P4 resnet +0.020", h13["P4"]["resnet"]["mean"], 0.020, 0.0005),
    ("P4 median +0.004", h13["P4"]["mlp"]["median"], 0.004, 0.0005),
    ("P5 mlp +0.018", h13["P5"]["mlp"]["mean"], 0.018, 0.0005),
    ("P5 resnet +0.019", h13["P5"]["resnet"]["mean"], 0.019, 0.0005),
    ("P5 mlp p 0.016", h13["P5"]["mlp"]["p"], 0.016, 0.0005),
    ("P5 resnet p 0.002", h13["P5"]["resnet"]["p"], 0.002, 0.0005),
    ("gaussianise alone p 0.37", h13["gaussianise_alone"]["mlp"]["p"], 0.37,
     0.005),
    ("retuned -0.057", h13["secondary_retuned_vs_frozen_mlp"]
     ["mean_retuned"], -0.057, 0.0005),
    ("retuned vs frozen p 0.049", h13["secondary_retuned_vs_frozen_mlp"]
     ["p_two_sided"], 0.049, 0.0005),
    ("ftt A -0.028", h11["ftt"]["noise4x"]["mean_delta"], -0.028, 0.0005),
    ("resnet A -0.062", h11["resnet"]["noise4x"]["mean_delta"], -0.062,
     0.0005),
]
for name, actual, expected, tol in b_claims:
    check(name, abs(actual - expected) <= tol, f"actual {actual:.5f}")
for key, net, n_ok, n in (("P1", "mlp", 55, 61), ("P1", "resnet", 56, 61),
                          ("P1_new", "mlp", 34, 37),
                          ("P1_new", "resnet", 33, 37),
                          ("P4", "mlp", 39, 51), ("P4", "resnet", 41, 51),
                          ("P5", "mlp", 17, 25), ("P5", "resnet", 20, 25)):
    e = h13[key][net]
    check(f"{key} {net} {n_ok}/{n}",
          e["n_direction_ok"] == n_ok and e["n"] == n,
          f"actual {e['n_direction_ok']}/{e['n']}")
check("ftt A 17/24", h11["ftt"]["noise4x"]["n_direction_ok"] == 17, "")
check("only dataset 38 incomplete", h13["incomplete_datasets"] == [38], "")
gh = h13["global_holm_all_primary"]
check("global Holm over 12 completed tests, all < 0.05",
      len(gh) == 12 and all(v < 0.05 for v in gh.values()), f"{gh}")
check("global Holm max MLP-like 0.032",
      abs(max(v for k, v in gh.items() if k.split("/")[1] != "ftt")
          - 0.032) < 0.0005, "")
check("global Holm max FTT 0.047",
      abs(max(v for k, v in gh.items() if k.endswith("/ftt")) - 0.047)
      < 0.0005, "")
f1, f4 = h13["P1"]["ftt"], h13["P4"]["ftt"]
for name, actual, expected, tol in (
        ("FTT P1 -0.020", f1["mean"], -0.020, 0.0005),
        ("FTT P1 median -0.003", f1["median"], -0.003, 0.0005),
        ("FTT P1 p 0.010", f1["p"], 0.010, 0.0005),
        ("FTT P1 new p 0.15", h13["P1_new"]["ftt"]["p"], 0.15, 0.005),
        ("FTT P4 +0.006", f4["mean"], 0.006, 0.0005),
        ("FTT P4 p 0.047", f4["p"], 0.047, 0.0005)):
    check(name, abs(actual - expected) <= tol, f"actual {actual:.5f}")
check("FTT P1 26/38", f1["n_direction_ok"] == 26 and f1["n"] == 38, "")
check("P1 medians -0.020 / -0.028",
      abs(h13["P1"]["mlp"]["median"] + 0.020) < 0.0005
      and abs(h13["P1"]["resnet"]["median"] + 0.028) < 0.0005, "")
check("FTT P1 new n=16", h13["P1_new"]["ftt"]["n"] == 16, "")
check("FTT P4 26/42", f4["n_direction_ok"] == 26 and f4["n"] == 42, "")
check("FTT P2 flagged incomplete", h13["P2"]["ftt"].get("incomplete"), "")

# --- I. exploratory natural weak-feature link (h14) ------------------------
h14 = json.loads((RESULTS_DIR / "h14_report.json").read_text())
n_claims = [
    ("weak fraction median 11%", h14["weak_fraction_median"], 0.11, 0.006),
    ("E1 mlp rho -0.24", h14["E1"]["mlp"]["rho"], -0.24, 0.005),
    ("E1 mlp p 0.058", h14["E1"]["mlp"]["p_two_sided"], 0.058, 0.0005),
    ("E1 resnet rho -0.31", h14["E1"]["resnet"]["rho"], -0.31, 0.005),
    ("E1 resnet p 0.015", h14["E1"]["resnet"]["p_two_sided"], 0.015,
     0.0005),
]
for name, actual, expected, tol in n_claims:
    check(name, abs(actual - expected) <= tol, f"actual {actual:.5f}")
check("15 datasets with >=25% weak features",
      h14["n_datasets_with_weak_ge_25pct"] == 15, "")
check("E2 |rho| < 0.07", all(abs(h14["E2"][n]["rho"]) < 0.07
                              for n in ("mlp", "resnet")), "")
for banned in ("pre-registered", "drive the gap", "our original",
               "original analysis"):
    check(f"banned term absent: {banned}", banned not in tex_now.lower(), "")
for banned in ("coin-flip", "surgery", "margin map", "causal converse",
               "causally aligned", "reviewer"):
    check(f"banned term absent: {banned}", banned not in tex_now.lower(), "")

# --- F. bibliography integrity ---------------------------------------------
tex = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
bib = (ROOT / "paper" / "refs.bib").read_text(encoding="utf-8")
cited = set()
for m in re.findall(r"\\cite[tp]?\{([^}]*)\}", tex):
    cited.update(k.strip() for k in m.split(","))
bibkeys = set(re.findall(r"@\w+\{([^,]+),", bib))
check("no missing bib entries", cited <= bibkeys,
      f"missing: {cited - bibkeys}")
check("no orphaned bib entries", bibkeys <= cited,
      f"orphans: {bibkeys - cited}")
check("no duplicate bib keys",
      len(re.findall(r"@\w+\{", bib)) == len(bibkeys), "")

audit = {"failures": failures,
         "noise_decomposition": {"mlp": float(nm), "gbdt": float(ng)},
         "rescue_decomposition": {"mlp": float(rm), "gbdt": float(rg)},
         "single_descriptor_top3": best,
         "mean_decided_margin": mean_dec,
         "descriptor_count": ndesc,
         "top5_sizes": [(int(d), int(sizes[d])) for d in top5]}
(RESULTS_DIR / "audit_report.json").write_text(json.dumps(audit, indent=2))
print(f"\n{'ALL CLAIMS VERIFIED' if not failures else 'FAILURES: ' + str(failures)}")
sys.exit(1 if failures else 0)
