"""Quick inspection of the H1 meta-report: importances + extreme margins."""
import json

r = json.loads(open("d:/pub2/results/meta_report.json").read())
print("base rate nn_wins:", round(r["base_rate_nn_wins"], 3))
print("TOP-8 descriptors (permutation importance):")
for f in r["feature_importance"][:8]:
    print(f"  {f['feature']:22s} {f['mean']:+.4f} +- {f['std']:.4f}")
print()
worst = sorted(r["per_dataset"], key=lambda d: -abs(d["true_margin"]))[:6]
print("Largest |nn-gbdt| margins:")
for d in worst:
    print(f"  {d['name'][:28]:28s} true={d['true_margin']:+.4f} "
          f"pred={d['pred_margin']:+.4f} p_nn={d['p_nn_wins']:.2f}")
near = sum(1 for d in r["per_dataset"] if abs(d["true_margin"]) < 0.005)
print(f"\nnear-tie datasets (|margin|<0.005): {near}/61")
