"""Count-based confidence sweep for the hydromorphology detector.

Per image & class: TP = min(pred, gt), FP = pred - TP, FN = gt - TP (count matching,
no box IoU). Thresholds fitted on VALID, reported on TEST, final thresholds refit on ALL.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

D = Path(__file__).parent
AB = {"BP": "Bank_protection", "ER": "Erosion", "IS": "Island", "BI": "bifurcation",
      "BU": "building", "FT": "fallen_tree", "LB": "longitudinal_bar", "OB": "organic_bed",
      "RF": "riffle", "RD": "road", "RT": "root", "WS": "wood_stump", "WD": "woody_debris_jam"}
CLASSES = list(AB)
TH = [round(0.20 + 0.05 * i, 2) for i in range(13)]  # 0.20 .. 0.80

gt, split, preds = {}, {}, {}
for line in (D / "gt.txt").read_text().splitlines():
    if not line or line.startswith("#"):
        continue
    f = line.split()
    idx, sp, toks = int(f[0]), f[2], f[3:]
    if toks == ["EXCLUDE"]:
        continue
    split[idx] = sp
    gt[idx] = defaultdict(int)
    for t in toks:
        k = t.rstrip("0123456789")
        gt[idx][k] += int(t[len(k):])
for line in (D / "preds.txt").read_text().splitlines():
    if not line or line.startswith("#"):
        continue
    f = line.split()
    preds[int(f[0])] = [(t[:2], float("0" + t[2:])) for t in f[1:]]

assert set(gt) == set(preds), set(gt) ^ set(preds)
tot = defaultdict(int)
for i in gt:
    tot[split[i]] += sum(gt[i].values())
print("GT boxes per split:", dict(tot), "| images:", {s: sum(1 for i in gt if split[i] == s) for s in ("valid", "test")})


def counts(ids, cls, th):
    tp = fp = fn = 0
    for i in ids:
        p = sum(1 for c, s in preds[i] if c == cls and s >= th - 1e-9)
        g = gt[i].get(cls, 0)
        m = min(p, g)
        tp, fp, fn = tp + m, fp + p - m, fn + g - m
    return tp, fp, fn


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def best_th(ids, cls):
    best = None
    for th in TH:
        f = prf(*counts(ids, cls, th))[2]
        if best is None or f > best[1] + 1e-9:
            best = (th, f)
    return best


def micro(ids, thmap):
    T = [0, 0, 0]
    for c in CLASSES:
        for k, v in enumerate(counts(ids, c, thmap[c])):
            T[k] += v
    return prf(*T)


valid = [i for i in gt if split[i] == "valid"]
test = [i for i in gt if split[i] == "test"]
alli = list(gt)

# global single threshold
glob = {}
for name, ids in (("valid", valid), ("all", alli)):
    rows = []
    for th in TH:
        rows.append((th, *micro(ids, {c: th for c in CLASSES})))
    glob[name] = rows
g_best_valid = max(glob["valid"], key=lambda r: r[3])
g_best_all = max(glob["all"], key=lambda r: r[3])

MIN_SUPPORT = 15
support_all = {c: sum(gt[i].get(c, 0) for i in alli) for c in CLASSES}
support_valid = {c: sum(gt[i].get(c, 0) for i in valid) for c in CLASSES}


def per_class(ids, support, fallback):
    out = {}
    for c in CLASSES:
        th, f = best_th(ids, c)
        out[c] = th if support[c] >= MIN_SUPPORT else fallback
    return out


pc_valid = per_class(valid, support_valid, g_best_valid[0])
pc_all = per_class(alli, support_all, g_best_all[0])

print(f"\nGlobal best threshold on VALID: {g_best_valid[0]}  P={g_best_valid[1]:.3f} R={g_best_valid[2]:.3f} F1={g_best_valid[3]:.3f}")
print("TEST with valid-fitted GLOBAL threshold:    P=%.3f R=%.3f F1=%.3f" % micro(test, {c: g_best_valid[0] for c in CLASSES}))
print("TEST with valid-fitted PER-CLASS thresholds: P=%.3f R=%.3f F1=%.3f" % micro(test, pc_valid))
print("TEST at default 0.40:                       P=%.3f R=%.3f F1=%.3f" % micro(test, {c: 0.40 for c in CLASSES}))
print("TEST at default 0.50:                       P=%.3f R=%.3f F1=%.3f" % micro(test, {c: 0.50 for c in CLASSES}))
print(f"\nGlobal best threshold on ALL 74 images: {g_best_all[0]}  F1={g_best_all[3]:.3f}")
print("ALL with final per-class thresholds:        P=%.3f R=%.3f F1=%.3f" % micro(alli, pc_all))

print("\nGlobal sweep (ALL):  th   P     R     F1")
for th, p, r, f in glob["all"]:
    print(f"                    {th:.2f} {p:.3f} {r:.3f} {f:.3f}")

print("\nPer-class (ALL 74 images), final threshold and count-matching metrics at it:")
print(f"{'class':18s} {'GT':>4s} {'th':>5s} {'P':>6s} {'R':>6s} {'F1':>6s}  {'pred_total':>10s}")
final = {}
rows = []
for c in sorted(CLASSES, key=lambda c: -support_all[c]):
    th = pc_all[c]
    tp, fp, fn = counts(alli, c, th)
    p, r, f = prf(tp, fp, fn)
    final[AB[c]] = th
    rows.append({"class": AB[c], "gt_instances": support_all[c], "threshold": th,
                 "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3),
                 "predicted_total": tp + fp, "per_class_fit": support_all[c] >= MIN_SUPPORT})
    flag = "" if support_all[c] >= MIN_SUPPORT else "  (low support -> global)"
    print(f"{AB[c]:18s} {support_all[c]:4d} {th:5.2f} {p:6.3f} {r:6.3f} {f:6.3f}  {tp+fp:10d}{flag}")

out = {"method": "count-matching per image/class on 74 held-out images (valid 49 + test 25)",
       "global_threshold": g_best_all[0], "class_thresholds": final, "per_class": rows,
       "test_holdout_f1_per_class_thresholds": round(micro(test, pc_valid)[2], 3),
       "test_holdout_f1_global_threshold": round(micro(test, {c: g_best_valid[0] for c in CLASSES})[2], 3)}
(D / "thresholds.json").write_text(json.dumps(out, indent=2))
print("\n" + json.dumps(final))
