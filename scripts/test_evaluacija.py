# Provere za Nedelju 5 (BM25 i evaluacija). Pokretanje iz korena projekta:
#   python -m scripts.test_evaluacija
# Nezavisan racun ide kroz SVE pesme iz tokeni (set sme samo u testu).

import inspect, math, random, re, time
from collections import Counter

from mrvectorbeat import pipeline, tekst, rang, evaluacija as ev
from mrvectorbeat.bm25 import Bm25Model

df = pipeline.ucitaj_katalog()
tok = pipeline.ucitaj_tokene()
ind = pipeline.ucitaj_indeks()
tfidf = pipeline.ucitaj_tfidf()
bm25 = pipeline.ucitaj_bm25()
fam = df["porodica"].to_numpy()
ids = list(df["track_id"])
N = len(ids)
TF = [Counter(tok[t]) for t in ids]
DL = [len(tok[t]) for t in ids]
AVG = sum(DL) / N
DF = Counter(t for c in TF for t in c)


def provera(opis, uslov):
    print(f"{'OK ' if uslov else 'GRESKA'} {opis}")


def brute_bm25(upit, k1, b):
    q = set(tekst.termini(upit))
    out = {}
    for d in range(N):
        if any(t in TF[d] for t in q):
            out[d] = sum(math.log10(1 + (N - DF[t] + 0.5) / (DF[t] + 0.5)) * TF[d][t] * (k1 + 1)
                         / (TF[d][t] + k1 * (1 - b + b * DL[d] / AVG)) for t in q if t in TF[d])
    return out


def isto(x, y, tol=1e-9):
    return x.keys() == y.keys() and all(abs(x[d] - y[d]) < tol for d in x)


# --- BM25 -----------------------------------------------------------------
print("BM25:")
provera(f"idf >= 0 za svih {len(bm25.idf)} termina (najmanji {min(bm25.idf.values()):.4f})",
        all(v >= 0 for v in bm25.idf.values()))
provera("duzina pesme == broj termina iz tokeni, avgdl tacan",
        list(bm25.duzine) == DL and abs(bm25.avgdl - AVG) < 1e-9)
random.seed(5)
recnik = list(ind.recnik)
upiti = [" ".join(random.sample(recnik, random.randint(1, 6))) for _ in range(40)] + ["love heart", "love love"]
ok = True
for u in upiti:
    ok &= isto(bm25.ocene(ind, u), brute_bm25(u, bm25.k1, bm25.b))
provera(f"BM25 == nezavisan racun kroz svih {N} pesama ({len(upiti)} upita, k1={bm25.k1}, b={bm25.b})", ok)
ok = all(isto(bm25.ocene(ind, "love heart night", "bm25", k1, b), brute_bm25("love heart night", k1, b))
         for k1, b in ((0.0, 0.5), (2.0, 0.0), (1.2, 1.0)))
provera("BM25 sa zadatim k1, b == nezavisan racun (k1 = 0, b = 0, b = 1)", ok)
provera(f"ocenjeno samo kandidata: {len(bm25.ocene(ind, 'love heart'))} od {N} (love heart)",
        len(bm25.ocene(ind, "love heart")) < N)
for mera in ("tfidf", "kosinus", "bm25"):
    prazno = [rang.rangiraj(ind, tfidf, bm25, u, mera, fam)[0] for u in ("", "the you and", "qqqzzz")]
    provera(f"{mera}: prazan upit, samo stop-reci, nepostojeca rec -> [] (bez pada)", prazno == [[], [], []])
    provera(f"{mera}: nepostojeca rec ne menja ostale ocene",
            (bm25 if mera == "bm25" else tfidf).ocene(ind, "love qqqzzz", mera)
            == (bm25 if mera == "bm25" else tfidf).ocene(ind, "love", mera))
izvor = inspect.getsource(Bm25Model.ocene)
provera("grep: u Bm25Model.ocene nema log ni sum (idf i duzine se samo citaju)",
        "log" not in izvor and "sum(" not in izvor)
try:
    tfidf.ocene(ind, "love", "bm25")
    provera("TF-IDF model odbija meru bm25", False)
except ValueError:
    provera("TF-IDF model odbija meru bm25 (ne racuna je tiho kao kosinus)", True)

# --- mere -------------------------------------------------------------------
print("Mere (rucno izracunati primeri):")
lista = ["a", "b", "c", "d", "e", "f"]
oc = {"b": 2, "e": 1, "x": 2}                     # x je relevantna, ali je lista nije vratila
provera("P@5 = 1/5 (samo b je pogodak sa pragom 2)", abs(ev.p_na_5(lista, oc, 2) - 0.2) < 1e-12)
provera("RR = 1/2 (prvi pogodak na mestu 2)", abs(ev.reciprocni_rang(lista, oc, 2) - 0.5) < 1e-12)
provera("AP = (1/2) / 2 = 0.25: deli brojem SVIH relevantnih (b i x), ne pogodjenih (bilo bi 0.5)",
        abs(ev.prosecna_preciznost(lista, oc, 2) - 0.25) < 1e-12)
dcg = 2 / math.log2(3) + 1 / math.log2(6)
idcg = 2 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
provera("nDCG@5 = (2/log2 3 + 1/log2 6) / (2 + 2/log2 3 + 1/log2 4)",
        abs(ev.ndcg_na_5(lista, oc) - dcg / idcg) < 1e-12)
provera("nDCG razlikuje 2 i 1: ista lista, b ima ocenu 1 umesto 2 -> druga vrednost",
        ev.ndcg_na_5(lista, oc) != ev.ndcg_na_5(lista, {"b": 1, "e": 1, "x": 2}))
provera("tematski prag 1: e je pogodak, P@5 = 2/5", abs(ev.p_na_5(lista, oc, 1) - 0.4) < 1e-12)
provera("upit bez relevantnih -> sve mere 0",
        all(v == 0 for v in ev.mere_upita(lista, {}, 2).values()))
m = ev.mere_upita(["x", "b"], {"x": 2, "b": 2}, 2)
provera("savrsena lista od 2 relevantne: RR = AP = nDCG = 1, P@5 = 2/5",
        m["MRR"] == 1 and abs(m["MAP"] - 1) < 1e-12 and abs(m["nDCG@5"] - 1) < 1e-12 and m["P@5"] == 0.4)

# --- artefakt evaluacije ----------------------------------------------------
print("Artefakt evaluacije:")
e = pipeline.ucitaj_evaluaciju()
q = pipeline.ucitaj_qrels()
provera(f"upita: {len(e['upiti'])} (147 + 53 + 22)",
        [sum(u["grupa"] == g for u in e["upiti"]) for g in ev.GRUPE] == [147, 53, 22])
t0 = time.perf_counter()
ok = ok_por = True
for mera, po_upitu in e["liste"].items():
    for u in e["upiti"]:
        sada = [d for d, _ in rang.rangiraj(ind, tfidf, bm25, u["upit"], mera, fam)[0]]
        ok &= list(po_upitu[u["upit"]]) == sada
        ok_por &= len({fam[d] for d in sada}) == len(sada)
provera(f"sve rang-liste iz artefakta == ono sto strana Stihovi vraca sada "
        f"({3 * len(e['upiti'])} lista, {time.perf_counter() - t0:.1f} s)", ok)
provera("u svakoj listi od svake porodice najvise jedna pesma", ok_por)
provera(f"izabrani k1, b u artefaktu == model ({e['k1']}, {e['b']})", (e["k1"], e["b"]) == (bm25.k1, bm25.b))
mreza = {(k1, b): m for k1, b, m in e["mreza"]}
provera("mreza: za k1 = 0 MRR ne zavisi od b (b tada nema uticaja u formuli)",
        len({round(m, 12) for (k1, b), m in mreza.items() if k1 == 0}) == 1)
provera(f"mreza: {len({k for k, _ in mreza})} vrednosti k1 x {len({b for _, b in mreza})} vrednosti b",
        len({b for _, b in mreza}) >= 3)
ocene = pipeline.ocene_po_upitu(q)
bez = [u["upit"] for u in e["upiti"] if u["grupa"] == ev.GRUPE[0]]
mrr = sum(ev.reciprocni_rang([fam[d] for d in e["liste"]["bm25"][u]], ocene[u], 2) for u in bez) / len(bez)
provera(f"MRR BM25 iz artefakta ({mrr:.4f}) == tacka mreze za izabrane k1, b",
        abs(mrr - mreza[(bm25.k1, bm25.b)]) < 1e-12)
provera("tipfeler se ne ispravlja: upit sa greskom ide u model doslovno",
        all(list(e["liste"]["bm25"][u["upit"]]) == [d for d, _ in rang.uredi(bm25.ocene(ind, u["upit"]), fam)[0]]
            for u in e["upiti"] if u["tipfeler"]))
