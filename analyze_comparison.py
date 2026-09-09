"""Analyze comparison data and print top regressions/improvements."""
import dotenv; dotenv.load_dotenv(".env")
import os, sys
from agent_evals.reporting.opik_target import resolve_opik_url
url = resolve_opik_url()
os.environ.setdefault("OPIK_URL_OVERRIDE", url)
os.environ["OPIK_PROJECT_NAME"] = "Agents"
import opik
from generate_report import _make_client, _find_by_tag, _get_items, _score_map, _case_name, _credits

client = _make_client()
exps1 = _find_by_tag(client, "staging-eu-20260908-181220", "side1")
exps2 = _find_by_tag(client, "dev-weu-20260908-181220", "side2")
matched = sorted(set(exps1) & set(exps2))

rows = []
for name in matched:
    e1, e2 = exps1[name], exps2[name]
    items1 = _get_items(client, e1.id)
    items2 = _get_items(client, e2.id)
    by1 = {_case_name(it): it for it in items1}
    by2 = {_case_name(it): it for it in items2}
    for cn in sorted(set(by1) | set(by2)):
        it1, it2 = by1.get(cn), by2.get(cn)
        sm1, sm2 = _score_map(it1) if it1 else {}, _score_map(it2) if it2 else {}
        v1, r1 = sm1.get("overall", (None, ""))
        v2, r2 = sm2.get("overall", (None, ""))
        delta = (v2 - v1) if v1 is not None and v2 is not None else None
        try:
            c1 = float(_credits(it1)) if it1 else 0.0
        except (TypeError, ValueError):
            c1 = 0.0
        try:
            c2 = float(_credits(it2)) if it2 else 0.0
        except (TypeError, ValueError):
            c2 = 0.0
        rows.append((name, cn, v1, v2, delta, r1[:300], r2[:300], c1, c2, e1.id, e2.id))

rows.sort(key=lambda r: r[4] if r[4] is not None else 0)
print("=== TOP 20 REGRESSIONS ===")
for r in rows[:20]:
    print()
    print(f"{r[0]} / {r[1]}")
    print(f"  {r[2]} -> {r[3]}  delta={r[4]}")
    print(f"  credits: {r[7]:.4f} -> {r[8]:.4f}")
    if r[5]: print(f"  reason1: {r[5]}")
    if r[6]: print(f"  reason2: {r[6]}")

print()
print("=== TOP 15 IMPROVEMENTS ===")
for r in sorted(rows, key=lambda r: -(r[4] if r[4] is not None else 0))[:15]:
    print()
    print(f"{r[0]} / {r[1]}")
    print(f"  {r[2]} -> {r[3]}  delta={r[4]}")
    if r[5]: print(f"  reason1: {r[5]}")
    if r[6]: print(f"  reason2: {r[6]}")
