import pandas as pd
from mrvectorbeat.tekst import skloni_apostrofe
import re
from collections import Counter

def izvuci_godinu(tekst) :
    prvi = str(tekst).split('-')[0]
    if len(prvi) == 4 and prvi.isdigit() :
        return int(prvi)
    return pd.NA

def izvuci_godine(serija) :
    return pd.array([izvuci_godinu(d) for d in serija], dtype="Int64")

OZNAKE = {"remaster", "remastered", "live", "remix", "remixed", "mix", "edit", "edited", "version",
          "acoustic", "demo", "mono", "stereo", "deluxe", "instrumental", "radio", "extended",
          "bonus", "single", "original", "reprise", "unplugged", "session", "cover",
          "explicit", "clean", "anniversary", "edition", "recorded", "sped", "slowed"}
POCETNE = {"feat", "ft", "featuring", "with", "from"}
_DELOVI = re.compile(r"\s-\s|[()\[\]]")

def norm(s) :
    s = skloni_apostrofe(str(s).lower())
    return "".join(c if c.isalnum() else " " for c in s).split()

def je_oznaka(deo) :
    reci = norm(deo)
    if not reci :
        return False
    if reci[0] in POCETNE :
        return True
    return any(r in OZNAKE or (len(r) == 4 and r.isdigit() and r[:2] in ("19", "20")) for r in reci)

def kljuc_porodice(naslov, izvodjac) :
    delovi = [d for d in _DELOVI.split(str(naslov)) if d.strip()]
    if not delovi :
        delovi = [str(naslov)]
    osnova = [delovi[0]] + [d for d in delovi[1:] if not je_oznaka(d)]
    reci = norm(" ".join(osnova))
    for i, r in enumerate(reci) :
        if i > 0 and r in {"feat", "ft", "featuring"} :
            reci = reci[:i]
            break
    return " ".join(norm(izvodjac)) + "||" + " ".join(reci)

def pregled(df, n=30):
    odbaceno, zadrzano = Counter(), Counter()
    for naslov in df["track_name"]:
        delovi = [d.strip() for d in _DELOVI.split(str(naslov)) if d.strip()]
        for d in delovi[1:]:
            (odbaceno if je_oznaka(d) else zadrzano)[d.lower()] += 1
    print("NAJCESCE ODBACENO:", odbaceno.most_common(n))
    print("NAJCESCE ZADRZANO:", zadrzano.most_common(n))
    print("NAJVECE PORODICE:\n", df.groupby("porodica").size().sort_values(ascending=False).head(n))