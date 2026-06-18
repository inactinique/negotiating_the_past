#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export IRaMuTeQ — fichier 3 : les prompts « yes » ET les prompts qui partagent
une SESSION avec un prompt « yes » (donc aussi des voisins « no »).

Définition de « session » (identique à la section 7 du notebook) :
  même canal Discord + écart temporel <= 30 min + similarité textuelle >= 85
  (ratio de Levenshtein, rapidfuzz), via une fenêtre glissante de 5 voisins.
On garde toutes les sessions contenant au moins un « yes », et l'on exporte
TOUS leurs membres (yes = pivot, no = voisin), plus les « yes » isolés.

Un prompt = un texte. Ligne étoilée IRaMuTeQ avec variables :
  *classe_yes / *classe_no      -> classification du prompt lui-même
  *role_pivot / *role_voisin    -> pivot = yes ; voisin = no partageant une session avec un yes
  *session_reroll/_ecriture/_isole -> nature de la session (texte répété / réécrit / isolé)

PLAFOND / ÉCHANTILLONNAGE : voir export_iramuteq_justifs.py (réservoir, algorithme R).
  CAP = 100_000 par défaut ; None pour tout exporter.
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from export_iramuteq_justifs import clean_body, starred_line, ReservoirWriter

DATA_PATH = "full_results.csv"
OUT_DIR = Path("iramuteq"); OUT_DIR.mkdir(exist_ok=True)
OUT_NAME = "corpus_prompts_yes_sessions.txt"
CAP = 100_000          # <-- plafond ; None pour tout exporter
SEED = 42
CHUNK = 1_000_000

# paramètres de session (identiques au notebook section 7)
W, THRESH, DT_MAX, BATCH = 5, 85.0, 30 * 60, 1_000_000

spaced_digits_re = re.compile(r"(?<=\d) (?=\d)")
ws_re = re.compile(r"\s+")

# ------------------------------------------------------------------ chargement
print("chargement des caches et des colonnes…", flush=True)
row_map = pd.read_parquet("row_map.parquet")["prompt_row"].to_numpy()
meta = pd.read_parquet("metadata.parquet")
safe = np.clip(row_map, 0, None)
valid = row_map >= 0
ts = np.where(valid, meta["timestamp"].to_numpy("float64")[safe], np.nan)
ch = np.where(valid, meta["channel_id"].to_numpy("float64")[safe], np.nan)
del meta

yes_parts, prompt_parts = [], []
for chunk in pd.read_csv(DATA_PATH, usecols=["prompt", "references_past"], chunksize=CHUNK):
    yes_parts.append((chunk["references_past"].astype(str).str.strip().str.lower() == "yes").to_numpy())
    prompt_parts.append(chunk["prompt"].astype("string"))
yes_mask = np.concatenate(yes_parts)
prompts_all = pd.concat(prompt_parts, ignore_index=True)
n_total = len(prompts_all)
print(f"{n_total:,} lignes ; {int(yes_mask.sum()):,} « yes »", flush=True)

def norm_batch(rows):
    s = prompts_all.iloc[rows].astype(str)
    return (s.str.lower()
             .str.replace(spaced_digits_re, "", regex=True)
             .str.replace(ws_re, " ", regex=True).str.strip()).tolist()

# ------------------------------------------------------ détection des sessions
idx_sorted = np.lexsort((ts, ch))
ch_s, ts_s = ch[idx_sorted], ts[idx_sorted]
pa, pb = [], []
for k in range(1, W + 1):
    dt = ts_s[k:] - ts_s[:-k]
    cand = np.nonzero((ch_s[k:] == ch_s[:-k]) & (dt >= 0) & (dt <= DT_MAX))[0]
    for s0 in range(0, len(cand), BATCH):
        v = cand[s0:s0 + BATCH]
        sc = cpdist(norm_batch(idx_sorted[v]), norm_batch(idx_sorted[v + k]),
                    scorer=fuzz.ratio, workers=-1, score_cutoff=THRESH)
        hit = np.nonzero(sc)[0]
        pa.append(v[hit]); pb.append(v[hit] + k)
    print(f"  k={k} : {len(cand):,} candidats", flush=True)
pa, pb = np.concatenate(pa), np.concatenate(pb)
print(f"{len(pa):,} paires similaires", flush=True)

# union-find -> sessions (positions dans l'ordre trié)
parent = {}
def find(x):
    while parent.setdefault(x, x) != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x
for a, b in zip(pa, pb):
    ra, rb = find(int(a)), find(int(b))
    if ra != rb:
        parent[rb] = ra
groups = {}
for x in parent:
    groups.setdefault(find(x), []).append(x)
groups = [np.array(sorted(v)) for v in groups.values() if len(v) >= 2]

# on ne garde que les sessions contenant au moins un « yes »
kept = [m for m in groups if yes_mask[idx_sorted[m]].any()]
print(f"{len(groups):,} sessions ; {len(kept):,} contenant ≥ 1 « yes »", flush=True)

# ----------------------------------------- type de session + appartenance ligne
all_pos = np.concatenate(kept) if kept else np.array([], dtype=int)
member_rows = idx_sorted[all_pos]
member_norm = norm_batch(member_rows)
row_to_sid, row_to_type = {}, {}
off = 0
for sid, m in enumerate(kept):
    rows = idx_sorted[m]
    norms = member_norm[off:off + len(m)]; off += len(m)
    stype = "reroll" if len(set(norms)) < 2 else "ecriture"
    for r in rows:
        row_to_sid[int(r)] = sid
        row_to_type[int(r)] = stype

# corpus = membres des sessions « yes » ∪ tous les « yes » (les yes isolés inclus)
included = set(member_rows.tolist()) | set(np.nonzero(yes_mask)[0].tolist())
included = np.array(sorted(included))
print(f"{len(included):,} prompts dans le corpus (yes pivots + voisins + yes isolés)", flush=True)

# ------------------------------------------------------------------- écriture
writer = ReservoirWriter(OUT_DIR / OUT_NAME, CAP, SEED)
membership = []  # cache réutilisable
raw = prompts_all.iloc[included].astype(str).to_numpy()
for r, prompt in zip(included, raw):
    body = clean_body(spaced_digits_re.sub("", prompt))   # recolle « 1 9 2 0 » -> « 1920 »
    if len(body) < 3:
        continue
    is_yes = bool(yes_mask[r])
    stype = row_to_type.get(int(r), "isole")
    classe = "yes" if is_yes else "no"
    role = "pivot" if is_yes else "voisin"
    star = starred_line(classe=classe, role=role, session=stype)
    writer.add(f"{star}\n{body}\n\n")
    membership.append((int(r), row_to_sid.get(int(r), -1), stype, classe, role))

total_seen, _ = writer.close()
n_out = CAP if (CAP and total_seen > CAP) else total_seen
print(f"{total_seen:,} prompts non vides ; {n_out:,} écrits dans {OUT_DIR / OUT_NAME} (cap={CAP})")

pd.DataFrame(membership, columns=["fr_row", "session_id", "session_type", "classe", "role"]) \
  .to_parquet("iramuteq_prompts_membership.parquet")
print("appartenance sauvegardée : iramuteq_prompts_membership.parquet")
