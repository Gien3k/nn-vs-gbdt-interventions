"""Compact verdict summary for H6-H9."""
import json

p = "d:/pub2/results/"
h6 = json.loads(open(p + "h6_report.json").read())
for coh in ("pilot", "extension", "pooled"):
    print("H6", coh + ":")
    for k, v in h6[coh].items():
        if v:
            print(f"  {k:15s} n={v['n_datasets']:2d} "
                  f"delta={v['mean_delta_at_max']:+.4f} "
                  f"dir={v['n_direction_ok']}/{v['n_datasets']} "
                  f"holm_p={v.get('holm_p'):.4f} "
                  f"conf={v.get('confirmed_at_0.05')}")
h7 = json.loads(open(p + "h7_report.json").read())
print("H7:", {k: (round(v, 4) if isinstance(v, float) else v)
              for k, v in h7.items()
              if k not in ("per_dataset", "madelon_spotlight")})
print("H7 madelon:", h7["madelon_spotlight"])
h8 = json.loads(open(p + "h8_report.json").read())
print("H8:", {k: (round(v, 4) if isinstance(v, float) else v)
              for k, v in h8.items() if k != "per_dataset"})
h9 = json.loads(open(p + "h9_report.json").read())
print("H9 primary:", h9["primary_extension_replication"])
print("H9 select_features:", h9["exploratory_select_features"])
