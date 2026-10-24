# scripts/test_indeks.py
#
# Provera Nedelje 3. Pokretanje iz korena projekta (posle scripts.pripremi):
#   python -m scripts.test_indeks
#
# Rezultati indeksa se porede sa obicnim prolaskom kroz sve pesme. set se
# ovde koristi samo kao nezavisna kontrola - u kodu upita ga nema.

import gzip, pickle, random

from mrvectorbeat import pipeline, tekst

df = pipeline.ucitaj_katalog()
tok, poz = pipeline.ucitaj_tokene(), pipeline.ucitaj_pozicije()
ind = pipeline.ucitaj_indeks()
ids = list(df["track_id"])
T = [tok[t] for t in ids]
P = [poz[t] for t in ids]


def provera(opis, uslov):
    print(f"{'OK ' if uslov else 'GRESKA'} {opis}")


# --- Tokeni i pozicije --------------------------------------------------

stari = pipeline.ART / "tokeni_stari.pkl.gz"
if stari.exists():
    with gzip.open(stari, "rb") as f:
        provera("tokeni.pkl.gz == tokeni_stari.pkl.gz (ucitan sadrzaj)", pickle.load(f) == tok)
else:
    print("-- nema tokeni_stari.pkl.gz, poredjenje preskoceno")
provera(f"tokena {sum(map(len, T))} == 3360835", sum(map(len, T)) == 3360835)
provera("len(pozicije) == len(tokeni) za svaku pesmu", all(len(p) == len(t) for t, p in zip(T, P)))
provera("termini na pozicijama == tekst.termini",
        all(tekst.termini_sa_pozicijama(s) == (T[d], list(P[d])) for d, s in enumerate(df["lyrics"][:2000])))
provera("ind.track_ids == katalog", ind.track_ids == ids)

# --- Isecak: broj reci sa lower() i bez njega ------------------------------

razlicite = [d for d, s in enumerate(df["lyrics"])
             if len(tekst.izdvajanje_reci(s)) != len(tekst.izdvajanje_reci(s.lower()))]
provera(f"izdvajanje_reci: isti broj reci sa lower() i bez, za svih {len(df)} pesama "
        f"(razlicitih: {len(razlicite)})", not razlicite)

# --- Nasumicni upiti naspram prolaska kroz sve pesme ------------------------

mapa = [dict(zip(p, t)) for t, p in zip(T, P)]        # pozicija -> termin
sadrzi = [set(t) for t in T]
random.seed(7)
recnik = list(ind.recnik)

ok = True
for _ in range(150):
    upit = " ".join(random.sample(recnik, random.randint(1, 3)))
    zabr = " ".join(random.sample(recnik, 2))
    q, z = tekst.termini(upit), tekst.termini(zabr)   # isti put kao u indeksu
    if not q:
        continue
    ok &= ind.bulov("I", upit) == [d for d in range(ind.n) if all(x in sadrzi[d] for x in q)]
    ok &= ind.bulov("ILI", upit) == [d for d in range(ind.n) if any(x in sadrzi[d] for x in q)]
    ok &= ind.bulov("NE", upit, zabr) == [d for d in range(ind.n) if all(x in sadrzi[d] for x in q)
                                          and not any(x in sadrzi[d] for x in z)]
    ok &= ind.bulov("NE", "", zabr) == [d for d in range(ind.n) if not any(x in sadrzi[d] for x in z)]
provera("Bulov I, ILI, NE (150 upita)", ok)

ok = True
for _ in range(150):
    d0 = random.randrange(ind.n)
    if len(P[d0]) < 6:
        continue
    i = random.randrange(len(P[d0]) - 3)
    reci = tekst.izdvajanje_reci(df["lyrics"][d0])
    fraza = " ".join(reci[P[d0][i]:P[d0][i + random.randint(0, 2)] + 1])    # fraza iz stvarnog stiha
    koreni, pz = tekst.termini_sa_pozicijama(fraza)
    pom = [p - pz[0] for p in pz]
    ocek = []
    for d in range(ind.n):
        s = [p for p in P[d] if all(mapa[d].get(p + m) == k for k, m in zip(koreni, pom))]
        if s:
            ocek.append((d, s, pom[-1] + 1))
    ok &= ind.frazni(fraza) == ocek and any(r[0] == d0 for r in ocek)
provera("Frazni (150 fraza iz stihova)", ok)

ok = True
for _ in range(100):
    a, b = random.sample(recnik[:400], 2)
    if random.random() < 0.2:
        b = a
    if tekst.termini(a) != [a] or tekst.termini(b) != [b]:
        continue
    k = random.randint(1, 6)
    ocek = []
    for d in range(ind.n):
        pa = [p for p in P[d] if mapa[d][p] == a]
        pb = [p for p in P[d] if mapa[d][p] == b]
        par = [(p, q) for p in pa for q in pb if abs(p - q) < k and (a != b or p < q)]
        if par:
            ocek.append((d, par))
    ok &= ind.blizina(a, b, k) == ocek
provera("Blizina (100 upita, oko 20% ista rec)", ok)

# --- Granicni slucajevi ---------------------------------------------------

provera("prazan upit", ind.bulov("I", "") == ind.bulov("ILI", "") == ind.bulov("NE", "", "")
        == ind.frazni("") == ind.blizina("", "love", 3) == [])
provera("upit samo od stop-reci", ind.bulov("I", "it is the") == ind.frazni("it is the") == [])
provera("termin koji ne postoji", ind.bulov("I", "love qqqzzz") == ind.frazni("love qqqzzz") == []
        and ind.bulov("ILI", "love qqqzzz") == list(ind.postings("love"))
        and ind.bulov("NE", "love", "qqqzzz") == list(ind.postings("love")))
provera("jedan termin u frazi", [r[0] for r in ind.frazni("love")] == list(ind.postings("love")))
provera("fraza duza od svake pesme", ind.frazni(" ".join(["love"] * 5000)) == [])
provera("stop-reci na pocetku i kraju fraze", ind.frazni("the love of my life is") == ind.frazni("love of my life"))
provera("polje blizine sa vise termina", ind.blizina("love you baby", "heart", 3) == [])
