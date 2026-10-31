"""Persian clinical-note entity extraction with negation handling and ICD-10 suggestions for human review.

Lab prototype on SYNTHETIC notes only (TRL 4). Dictionary + normalisation + fuzzy matching; every code is a SUGGESTION.
"""
import re
import numpy as np

# (surface forms, type, ICD-10 code or None). Lexicon is a small illustrative subset.
LEXICON = [
    (["دیابت", "دیابت نوع ۲", "دیابت شیرین"], "condition", "E11"),
    (["فشار خون بالا", "پرفشاری خون", "هیپرتانسیون"], "condition", "I10"),
    (["آسم", "آسم برونشیال"], "condition", "J45"),
    (["پنومونی", "ذات الریه"], "condition", "J18"),
    (["نارسایی قلبی", "نارسایی احتقانی قلب"], "condition", "I50"),
    (["آپاندیسیت", "التهاب آپاندیس"], "condition", "K35"),
    (["سنگ کلیه"], "condition", "N20"),
    (["میگرن"], "condition", "G43"),
    (["کم خونی", "آنمی"], "condition", "D64"),
    (["سکته مغزی"], "condition", "I64"),
    (["تب"], "symptom", None), (["سرفه"], "symptom", None), (["تنگی نفس"], "symptom", None),
    (["درد قفسه سینه"], "symptom", None), (["سردرد"], "symptom", None), (["تهوع"], "symptom", None),
    (["متفورمین"], "drug", None), (["آتورواستاتین"], "drug", None), (["آملودیپین"], "drug", None),
    (["سالبوتامول"], "drug", None), (["آموکسی سیلین"], "drug", None), (["آسپرین"], "drug", None),
]
NEG_BEFORE = {"بدون", "فاقد", "نفی"}
NEG_AFTER = {"ندارد", "نداشت", "منفی", "نبود", "نیست", "رد"}
AMBIG_CODES = set()

CRITERIA = [
    ("entity_f1", ">=", 0.90, "entity extraction F1 (type + normalised text)"),
    ("negation_accuracy", ">=", 0.90, "negation status accuracy on matched entities"),
    ("icd_top1_accuracy", ">=", 0.90, "ICD-10 suggestion accuracy on true conditions"),
]

_AR = {"ي": "ی", "ك": "ک", "ۀ": "ه", "ة": "ه", "أ": "ا", "إ": "ا"}
_DIG = {c: str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")}
_DIG.update({c: str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")})


def normalize(s):
    s = "".join(_DIG.get(_AR.get(c, c), _AR.get(c, c)) for c in s)
    s = re.sub("[ً-ْـ]", "", s).replace("‌", " ")
    return re.sub(r"\s+", " ", s).strip()


def _lev1(a, b):
    """True if edit distance between a and b is <= 1."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


_INDEX = {}
for forms, typ, icd in LEXICON:
    for f in forms:
        _INDEX[normalize(f).replace(" ", "")] = (normalize(f), typ, icd)
_MAXN = max(len(normalize(f).split()) for forms, _, _ in LEXICON for f in forms)


def extract(text):
    toks = normalize(text).split(" ")
    used = [False] * len(toks); ents = []
    sent = []; sid = 0
    for tk in toks:  # sentence id per token: negation scope never crosses a full stop
        sent.append(sid)
        if tk.endswith((".", "؟", "؛", "!")):
            sid += 1
    for n in range(_MAXN, 0, -1):
        for i in range(len(toks) - n + 1):
            if any(used[i:i + n]):
                continue
            key = "".join(toks[i:i + n]).strip("،.:؛")
            hit = _INDEX.get(key); fuzzy = False
            if hit is None and len(key) >= 6:
                for k, v in _INDEX.items():
                    if _lev1(key, k):
                        hit = v; fuzzy = True; break
            if hit:
                for j in range(i, i + n):
                    used[j] = True
                norm, typ, icd = hit
                same = lambda j: sent[j] == sent[i]
                neg = any(t.strip("،.") in NEG_BEFORE for j, t in enumerate(toks[max(0, i - 3):i], max(0, i - 3)) if same(j)) or \
                    any(t.strip("،.") in NEG_AFTER for j, t in enumerate(toks[i + n:i + n + 3], i + n) if same(j))
                ents.append({"text": norm, "type": typ, "start": i, "negated": bool(neg), "icd10": icd if typ == "condition" else None,
                             "confidence": 0.8 if fuzzy else 0.95})
    return sorted(ents, key=lambda e: e["start"])


def suggest_codes(note_entities):
    """ICD-10 suggestions for affirmed conditions; low-confidence ones are flagged for human review."""
    out = []
    for e in note_entities:
        if e["type"] == "condition" and not e["negated"] and e["icd10"]:
            out.append({"code": e["icd10"], "from": e["text"], "confidence": e["confidence"], "needs_review": e["confidence"] < 0.9 or e["icd10"] in AMBIG_CODES})
    return out


def _typo(rng, s):
    if len(s) < 6 or rng.random() > 0.15:
        return s
    i = int(rng.integers(1, len(s) - 1))
    return s[:i] + s[i + 1:] if rng.random() < 0.5 else s[:i] + s[i] + s[i:]


def _variant(rng, s):
    if rng.random() < 0.4:
        s = s.replace("ی", "ي").replace("ک", "ك")
    if rng.random() < 0.3:
        s = s.replace(" ", "‌")
    return s


def simulate(seed=0, n=500):
    rng = np.random.default_rng(seed)
    byt = {t: [x for x in LEXICON if x[1] == t] for t in ("condition", "symptom", "drug")}
    notes = []
    for _ in range(n):
        age = int(rng.integers(5, 90)); parts = ["بیمار %d ساله" % age]; gold = []

        def add(entry, neg_style=None):
            forms, typ, icd = entry
            f = forms[int(rng.integers(0, len(forms)))]
            surf = _variant(rng, _typo(rng, f)) if typ != "drug" else _variant(rng, f)
            gold.append({"text": normalize(f), "type": typ, "negated": neg_style is not None, "icd10": icd if typ == "condition" else None, "fuzzy": surf != f and normalize(surf) != normalize(f)})
            return surf
        s = byt["symptom"][int(rng.integers(0, len(byt["symptom"])))]
        parts.append("با شکایت %s مراجعه کرد." % add(s))
        c = byt["condition"][int(rng.integers(0, len(byt["condition"])))]
        parts.append("سابقه %s دارد." % add(c))
        if rng.random() < 0.6:
            s2 = byt["symptom"][int(rng.integers(0, len(byt["symptom"])))]
            if gold and normalize(s2[0][0]) != gold[0]["text"]:
                style = rng.random()
                if style < 0.5:
                    parts.append("بدون %s." % add(s2, "pre"))
                else:
                    parts.append("%s ندارد." % add(s2, "post"))
        if rng.random() < 0.6:
            d = byt["drug"][int(rng.integers(0, len(byt["drug"])))]
            parts.append("داروی %s تجویز شد." % add(d))
        notes.append({"text": " ".join(parts), "gold": gold})
    return notes


def evaluate(seed=0):
    notes = simulate(seed)
    tp = fp = fn = 0; neg_ok = neg_n = 0; icd_ok = icd_n = 0
    for nt in notes:
        pred = extract(nt["text"]); pk = {(e["text"], e["type"]): e for e in pred}; gk = {(g["text"], g["type"]): g for g in nt["gold"]}
        tp += len(pk.keys() & gk.keys()); fp += len(pk.keys() - gk.keys()); fn += len(gk.keys() - pk.keys())
        for k in pk.keys() & gk.keys():
            neg_n += 1; neg_ok += pk[k]["negated"] == gk[k]["negated"]
        sug = {s["from"]: s["code"] for s in suggest_codes(pred)}
        for g in nt["gold"]:
            if g["type"] == "condition" and not g["negated"]:
                icd_n += 1; icd_ok += sug.get(g["text"]) == g["icd10"]
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return {"entity_f1": 2 * p * r / max(p + r, 1e-9), "entity_precision": p, "entity_recall": r,
            "negation_accuracy": neg_ok / max(neg_n, 1), "icd_top1_accuracy": icd_ok / max(icd_n, 1), "n_notes": len(notes)}


def generate(seed=0):
    notes = simulate(seed, 200)
    return {"notes": [dict(note_id="N%04d" % i, text=n["text"]) for i, n in enumerate(notes)],
            "gold_entities": [dict(note_id="N%04d" % i, text=g["text"], type=g["type"], negated=int(g["negated"]), icd10=g["icd10"] or "") for i, n in enumerate(notes) for g in n["gold"]]}


def demo(seed=0):
    t = "بیمار ۶۲ ساله با شکایت تنگی نفس مراجعه کرد. سابقه ديابت دارد. تب ندارد. داروی متفورمین تجویز شد."
    e = extract(t)
    return "\n".join(str(x) for x in e) + "\nICD suggestions: " + str(suggest_codes(e))
