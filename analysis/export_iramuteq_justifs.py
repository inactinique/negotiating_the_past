#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export IRaMuTeQ — fichiers 1 et 2 : les justifications « no » et « yes ».

Format IRaMuTeQ (corpus texte) :
  - fichier texte UTF-8 (sans BOM) ;
  - chaque « texte » (ici : une justification) est précédé d'une LIGNE ÉTOILÉE
    qui commence en début de ligne par quatre astérisques, suivie de variables
    « étoilées » de la forme *variable_modalite (caractères [A-Za-z0-9_] seulement,
    pas d'espace ni d'accent dans les modalités) ;
  - le corps du texte suit sur la/les lignes suivantes jusqu'à la prochaine ligne
    étoilée. Les astérisques sont réservés : on les retire du corps.

Exemple d'un bloc :
  **** *classe_no *fictif_1 *moderne_1 *gentime_court
  the prompt describes a fictional surreal concept with no reference to the past

PLAFONNEMENT (cap) ET ÉCHANTILLONNAGE
  CAP = 100_000  -> chaque corpus est plafonné à 100 000 textes.
  CAP = None     -> aucun plafond : tout le corpus est exporté (le fichier « no »
                    fait alors ~10 M de textes / ~2 Go, voir l'avertissement plus bas).
  Méthode : ÉCHANTILLONNAGE PAR RÉSERVOIR (algorithme R, Vitter 1985).
    On parcourt le corpus une seule fois. On garde les CAP premiers textes ; pour le
    n-ième texte (n > CAP), on l'accepte avec probabilité CAP/n et il remplace alors
    un texte tiré au hasard dans le réservoir. À la fin, chaque texte vu a exactement
    la probabilité CAP/N d'être retenu — échantillon aléatoire uniforme, sans connaître
    le total N à l'avance, en mémoire O(CAP). Graine fixe -> tirage reproductible.
    NB : changer CAP retire un nouvel échantillon (les échantillons ne sont pas emboîtés).
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd

DATA_PATH = "full_results.csv"
OUT_DIR = Path("iramuteq"); OUT_DIR.mkdir(exist_ok=True)
CAP = 100_000          # <-- plafond par corpus ; None pour tout exporter
SEED = 42
CHUNK = 1_000_000

# --- nettoyage du corps de texte -------------------------------------------------
ctrl_re   = re.compile(r"[\x00-\x1f\x7f]")          # caractères de contrôle / sauts de ligne
star_re    = re.compile(r"\*+")                       # astérisques (réservés par IRaMuTeQ)
ws_re      = re.compile(r"\s+")
# marqueur initial des justifications : **NO**:, **[no]**:, **[yes/no]:** , etc.
prefix_re  = re.compile(
    r"^\s*\*{0,2}\[?\s*(?:yes\s*/\s*no|no|yes)\s*\]?\s*\*{0,2}\s*:?\s*\*{0,2}\s*\[?\s*",
    re.IGNORECASE)

def clean_body(text):
    t = ctrl_re.sub(" ", str(text))
    t = star_re.sub(" ", t)
    t = ws_re.sub(" ", t).strip()
    return t

# --- ligne étoilée ---------------------------------------------------------------
modality_re = re.compile(r"[^A-Za-z0-9_]+")
def modality(value):
    return modality_re.sub("_", str(value)).strip("_") or "na"
def starred_line(**variables):
    return "**** " + " ".join(f"*{k}_{modality(v)}" for k, v in variables.items())

# --- typologies thématiques (identiques aux sections 2 et 3 du notebook) ----------
THEMES_NO = {  # appliquées à la justification « no »
    "fictif":     r"fiction|fantasy|surreal|imaginar|myth|sci-?fi|futurist",
    "moderne":    r"modern|contemporary|present-day|current",
    "style":      r"stylistic|aesthetic|vintage|retro",
    "sanshisto":  r"no (?:reference|historical|specific)|without (?:any )?historical|lacks (?:any )?historical",
    "abstrait":   r"abstract|conceptual",
    "popculture": r"video ?game|anime|celebrit|pop[- ]?culture|movie|film|comic",
}
THEMES_YES = {  # appliquées à la justification « yes »
    "artistes":    r"historical artists?|art movements?|art nouveau|impressionis|renaissance|baroque|ukiyo-?e|romanticis|pre-?raphaelite|realis[mt]",
    "periodes":    r"historical periods?|\beras?\b|centur(?:y|ies)|medieval|victorian|ancient|antiquity",
    "figures":     r"historical figures?|real-?life figures?",
    "evenements":  r"historical events?|\bwars?\b|battle|revolution",
    "mytho":       r"mytholog|folklore|biblical|legend|deit(?:y|ies)|religious",
    "styleepoque": r"daguerr|vintage|retro|archival|old photograph|film still|sepia",
}

# --- réservoir (échantillon uniforme de taille CAP en un passage) -----------------
class ReservoirWriter:
    """Écrit un corpus IRaMuTeQ ; échantillonne par réservoir si cap est défini,
    sinon écrit chaque texte directement (flux), en mémoire O(1)."""
    def __init__(self, path, cap, seed):
        self.path, self.cap = path, cap
        self.rng = np.random.default_rng(seed)
        self.n = 0
        self.buf = [] if cap is not None else None
        self.fh = open(path, "w", encoding="utf-8") if cap is None else None

    def add(self, block):
        self.n += 1
        if self.cap is None:                      # pas de plafond : flux direct
            self.fh.write(block); return
        if len(self.buf) < self.cap:              # remplissage initial du réservoir
            self.buf.append(block)
        else:                                     # remplacement avec proba cap/n
            j = int(self.rng.integers(0, self.n))
            if j < self.cap:
                self.buf[j] = block

    def close(self):
        if self.cap is None:
            self.fh.close()
        else:
            with open(self.path, "w", encoding="utf-8") as f:
                f.writelines(self.buf)
        return self.n, (self.cap if (self.cap and self.n > self.cap) else self.n)


def gentime_cuts(sample_rows=1_000_000):
    """Terciles du temps de génération, estimés sur un échantillon — pour la variable
    *gentime_ (court / moyen / long)."""
    s = pd.read_csv(DATA_PATH, usecols=["generation_time"], nrows=sample_rows)["generation_time"]
    q1, q2 = s.quantile([1/3, 2/3])
    return float(q1), float(q2)

def gentime_label(x, q1, q2):
    if not np.isfinite(x): return "na"
    return "court" if x < q1 else ("moyen" if x < q2 else "long")


def export(which, themes, out_name):
    assert which in ("no", "yes")
    q1, q2 = gentime_cuts()
    theme_res = {k: re.compile(v, re.IGNORECASE) for k, v in themes.items()}
    writer = ReservoirWriter(OUT_DIR / out_name, CAP, SEED)
    kept_text = 0
    for chunk in pd.read_csv(DATA_PATH,
                             usecols=["references_past", "justification", "generation_time"],
                             chunksize=CHUNK):
        labels = chunk["references_past"].astype(str).str.strip().str.lower()
        sub = chunk[labels == which]
        for just, gt in zip(sub["justification"].astype(str), sub["generation_time"]):
            raison = prefix_re.sub("", just)         # on retire le marqueur **NO**:/**YES**:
            low = raison.lower()
            body = clean_body(raison)
            if len(body) < 3:
                continue
            kept_text += 1
            flags = {k: ("1" if r.search(low) else "0") for k, r in theme_res.items()}
            star = starred_line(classe=which, **flags, gentime=gentime_label(gt, q1, q2))
            writer.add(f"{star}\n{body}\n\n")
    total_seen, _ = writer.close()
    n_out = CAP if (CAP and total_seen > CAP) else total_seen
    print(f"[{which}] {total_seen:,} justifications non vides ; "
          f"{n_out:,} écrites dans {OUT_DIR / out_name} (cap={CAP})", flush=True)


if __name__ == "__main__":
    export("no",  THEMES_NO,  "corpus_explications_no.txt")
    export("yes", THEMES_YES, "corpus_explications_yes.txt")
    print("terminé.")
