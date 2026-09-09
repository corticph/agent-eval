"""Quick comparison summary for all three tag groups."""
import dotenv; dotenv.load_dotenv(".env")
import os
from generate_report import _make_client, _find_by_tags, _get_items, _score_map, _case_name, _credits, build_comparison

client = _make_client()

# Old run (non-dedalus, non-interview/pubmed)
old_staging = "staging-eu-20260908-181220"
old_dev = "dev-weu-20260908-181220"
# New run (interview + pubmed re-run)
new_staging = "staging-eu-20260908-190243"
new_dev = "dev-weu-20260908-190243"
# Dedalus
ded_staging = "staging-eu-20260908-190243"
ded_dev = "dev-weu-20260908-190243"

# Side 1 = staging (baseline), Side 2 = dev (under test)
# Merge old + new tags for the non-dedalus experiments
print("=== NON-DEDALUS (old + new pubmed/interview) ===")
exps1 = _find_by_tags(client, [old_staging, new_staging], "staging")
exps2 = _find_by_tags(client, [old_dev, new_dev], "dev")
print(f"staging: {len(exps1)} suites, dev: {len(exps2)} suites")
matched = sorted(set(exps1) & set(exps2))
print(f"matched: {len(matched)}")
rows, suites, only1, only2 = build_comparison(client, exps1, exps2)
scored = [r for r in rows if r.delta is not None]
m1 = sum(r.v1 for r in scored) / len(scored) if scored else 0
m2 = sum(r.v2 for r in scored) / len(scored) if scored else 0
r_count = sum(1 for r in scored if r.delta < 0)
i_count = sum(1 for r in scored if r.delta > 0)
u_count = sum(1 for r in scored if r.delta == 0)
tc1 = sum(s.credits1 for s in suites)
tc2 = sum(s.credits2 for s in suites)
print(f"Mean: {m1:.3f} -> {m2:.3f} ({r_count} regressed, {i_count} improved, {u_count} unchanged)")
print(f"Credits: {tc1:.4f} -> {tc2:.4f} ({(tc2-tc1)/tc1*100 if tc1>0 else 0:+.1f}%)")
if only1: print(f"Only staging: {sorted(only1)}")
if only2: print(f"Only dev: {sorted(only2)}")

# Top regressions
rows.sort(key=lambda r: r.delta if r.delta is not None else 0)
print("\nTop 10 regressions:")
for r in rows[:10]:
    r2 = (r.reason2 or "")[:150]
    print(f"  {r.exp_name}/{r.case_name}: {r.v1} -> {r.v2} ({r.delta:+.3f}) | {r2}")

print("\nTop 10 improvements:")
for r in sorted(rows, key=lambda r: -(r.delta if r.delta is not None else 0))[:10]:
    print(f"  {r.exp_name}/{r.case_name}: {r.v1} -> {r.v2} ({r.delta:+.3f})")

print("\n\n=== DEDALUS ===")
exps1d = _find_by_tags(client, [ded_staging], "staging-dedalus")
exps2d = _find_by_tags(client, [ded_dev], "dev-dedalus")
print(f"staging: {len(exps1d)} suites, dev: {len(exps2d)} suites")
matched_d = sorted(set(exps1d) & set(exps2d))
print(f"matched: {len(matched_d)}")
rows_d, suites_d, only1d, only2d = build_comparison(client, exps1d, exps2d)
scored_d = [r for r in rows_d if r.delta is not None]
m1d = sum(r.v1 for r in scored_d) / len(scored_d) if scored_d else 0
m2d = sum(r.v2 for r in scored_d) / len(scored_d) if scored_d else 0
r_count_d = sum(1 for r in scored_d if r.delta < 0)
i_count_d = sum(1 for r in scored_d if r.delta > 0)
u_count_d = sum(1 for r in scored_d if r.delta == 0)
tc1d = sum(s.credits1 for s in suites_d)
tc2d = sum(s.credits2 for s in suites_d)
print(f"Mean: {m1d:.3f} -> {m2d:.3f} ({r_count_d} regressed, {i_count_d} improved, {u_count_d} unchanged)")
print(f"Credits: {tc1d:.4f} -> {tc2d:.4f} ({(tc2d-tc1d)/tc1d*100 if tc1d>0 else 0:+.1f}%)")
if only1d: print(f"Only staging: {sorted(only1d)[:10]}")
if only2d: print(f"Only dev: {sorted(only2d)[:10]}")

rows_d.sort(key=lambda r: r.delta if r.delta is not None else 0)
print("\nTop 10 dedalus regressions:")
for r in rows_d[:10]:
    r2 = (r.reason2 or "")[:150]
    print(f"  {r.exp_name}/{r.case_name}: {r.v1} -> {r.v2} ({r.delta:+.3f}) | {r2}")

print("\nTop 10 dedalus improvements:")
for r in sorted(rows_d, key=lambda r: -(r.delta if r.delta is not None else 0))[:10]:
    print(f"  {r.exp_name}/{r.case_name}: {r.v1} -> {r.v2} ({r.delta:+.3f})")

# Combined totals
all_scored = scored + scored_d
combined_m1 = sum(r.v1 for r in all_scored) / len(all_scored) if all_scored else 0
combined_m2 = sum(r.v2 for r in all_scored) / len(all_scored) if all_scored else 0
combined_r = sum(1 for r in all_scored if r.delta < 0)
combined_i = sum(1 for r in all_scored if r.delta > 0)
combined_u = sum(1 for r in all_scored if r.delta == 0)
combined_c1 = tc1 + tc1d
combined_c2 = tc2 + tc2d
print(f"\n\n=== COMBINED TOTALS ===")
print(f"Mean: {combined_m1:.3f} -> {combined_m2:.3f} ({combined_r} regressed, {combined_i} improved, {combined_u} unchanged, {len(all_scored)} total)")
print(f"Credits: {combined_c1:.4f} -> {combined_c2:.4f} ({(combined_c2-combined_c1)/combined_c1*100 if combined_c1>0 else 0:+.1f}%)")
