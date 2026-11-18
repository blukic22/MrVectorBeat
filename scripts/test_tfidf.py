import math, random, time
from collections import Counter

from mrvectorbeat import pipeline, tekst

df = pipeline.ucitaj_katalog()
tok = pipeline.ucitaj_tokene()
ind = pipeline.ucitaj_indeks()
model = pipeline.ucitaj_tfidf()
ids = list(df["track_id"])
fam = df["porodica"].to_numpy()
N = len(ids)
TF = [Counter(tok[t]) for t in ids]

DF = Counter()
for c in TF:
    for t in c:
        DF[t] += 1
IDF = {t: math.log10(N / n) for t, n in DF.items()}
L = lambda tf: 1.0 + math.log10(tf)
NLNC = [math.sqrt(sum(L(tf) ** 2 for tf in c.values())) for c in TF]
NLTC = [math.sqrt(sum((L(tf) * IDF[t]) ** 2 for t, tf in c.items())) for c in TF]


def brute_tfidf(upit):
    q = set(tekst.termini(upit))
    return {d: sum(L(TF[d][t]) * IDF[t] for t in q if t in TF[d])
            for d in range(N) if any(t in TF[d] for t in q)}


def brute_kosinus(upit):
    qc = Counter(tekst.termini(upit))
    wq = {t: L(tf) * IDF[t] for t, tf in qc.items() if t in IDF}
    qn = math.sqrt(sum(w * w for w in wq.values()))
    out = {}
    for d in range(N):
        if any(t in TF[d] for t in wq):
            out[d] = sum(wq[t] / qn * L(TF[d][t]) / NLNC[d] for t in wq if t in TF[d]) if qn else 0.0
    return out


def brute_ltc(a, b):
    return sum(L(TF[a][t]) * IDF[t] / NLTC[a] * L(TF[b][t]) * IDF[t] / NLTC[b]
               for t in TF[a] if t in TF[b])


def isto(x, y, tol=1e-9):
    return x.keys() == y.keys() and all(abs(x[d] - y[d]) < tol for d in x)


def provera(opis, uslov):
    print(f"{'OK ' if uslov else 'GRESKA'} {opis}")


random.seed(11)
recnik = list(ind.recnik)

provera(f"idf == log10(N / df) za svih {len(recnik)} termina",
        all(abs(model.idf[t] - IDF[t]) < 1e-12 for t in recnik))
provera("norme lnc i ltc iz modela == nezavisan racun",
        all(abs(model.norma_lnc[d] - NLNC[d]) < 1e-9 and abs(model.norma_ltc[d] - NLTC[d]) < 1e-9
            for d in range(N) if TF[d]))

upiti = [" ".join(random.sample(recnik, random.randint(1, 4))) for _ in range(120)]
upiti += ["love", "love heart", "love love heart", "the love you"]
ok1 = ok2 = ok3 = True
for u in upiti:
    o1, o2 = model.ocene(ind, u, "tfidf"), model.ocene(ind, u, "kosinus")
    ok1 &= isto(o1, brute_tfidf(u))
    ok2 &= isto(o2, brute_kosinus(u))
    rez, n_ocenjeno = model.rangiraj(ind, u, "kosinus", fam)
    ocene = [o for _, o in rez]
    ok3 &= ocene == sorted(ocene, reverse=True) and len({fam[d] for d, _ in rez}) == len(rez)
provera(f"TF-IDF == nezavisan racun ({len(upiti)} upita)", ok1)
provera(f"kosinus lnc.ltc == nezavisan racun ({len(upiti)} upita)", ok2)
provera("rangirano: opadajuce, od svake porodice samo jedna pesma", ok3)

rez, n_ocenjeno = model.rangiraj(ind, "love heart", "tfidf")
provera(f"ocenjeno samo kandidata: {n_ocenjeno} od {N} pesama (love heart)",
        n_ocenjeno == len(brute_tfidf("love heart")) and n_ocenjeno < N)
kos = dict(model.rangiraj(ind, "love heart", "kosinus")[0])
provera("kosinus je u [0, 1]", all(-1e-9 <= v <= 1 + 1e-9 for v in kos.values()))
sirovo = dict(model.rangiraj(ind, "love heart", "tfidf", fam)[0])
provera("verzije: u listi nema dve pesme iste porodice (love heart)",
        len({fam[d] for d in sirovo}) == len(sirovo))

ok = ok_sam = ok_por = True
for d in random.sample(range(N), 12):
    if not TF[d]:
        continue
    sl = model.slicne(ind, d, 10, fam)
    ocek = sorted(((brute_ltc(d, p), p) for p in range(N) if p != d and fam[p] != fam[d] and
                   any(t in TF[p] for t in TF[d] if IDF[t] > 0)), key=lambda x: (-x[0], x[1]))
    videne, top = set(), []
    for o, p in ocek:
        if fam[p] in videne:
            continue
        videne.add(fam[p])
        top.append((p, o))
        if len(top) == 10:
            break
    ok &= [p for p, _ in sl] == [p for p, _ in top] and all(abs(a[1] - b[1]) < 1e-9 for a, b in zip(sl, top))
    ok_sam &= all(p != d for p, _ in sl)
    ok_por &= all(fam[p] != fam[d] for p, _ in sl)
provera("slicne pesme == nezavisan racun (12 pesama)", ok)
provera("slicne: nema same pesme", ok_sam)
provera("slicne: nema verzija iste pesme (iste porodice)", ok_por)
d = next(i for i in range(N) if TF[i])
provera("kosinus pesme sa samom sobom = 1", abs(model.doprinosi(d, d, 10)[0] - 1) < 1e-9)

ok_zbir = ok_sim = ok_top = True
for _ in range(30):
    a, b = random.sample(range(N), 2)
    if not TF[a] or not TF[b]:
        continue
    ukupno, tab = model.doprinosi(a, b, 10)
    ok_zbir &= abs(ukupno - brute_ltc(a, b)) < 1e-9
    ok_sim &= abs(ukupno - model.doprinosi(b, a, 10)[0]) < 1e-9
    puno = model.doprinosi(a, b, 10 ** 6)[1]
    ok_top &= abs(sum(r[4] for r in puno) - 1) < 1e-9 or not puno
    ok_top &= [r[3] for r in tab] == sorted((r[3] for r in tab), reverse=True) and len(tab) <= 10
    ok_top &= all(r[1] == TF[a][r[0]] and r[2] == TF[b][r[0]] for r in tab)
provera("doprinosi: zbir doprinosa == kosinus", ok_zbir)
provera("doprinosi: simetrija cos(a, b) == cos(b, a)", ok_sim)
provera("doprinosi: udeli daju 100%, najvise 10 reci, tf tacan", ok_top)

for mera in ("tfidf", "kosinus"):
    provera(f"{mera}: prazan upit, samo stop-reci, nepostojeci termin -> []",
            model.rangiraj(ind, "", mera)[0] == model.rangiraj(ind, "the you and", mera)[0]
            == model.rangiraj(ind, "qqqzzz", mera)[0] == [])
provera("nepostojeci termin ne smeta ostalima",
        model.ocene(ind, "love qqqzzz", "tfidf") == model.ocene(ind, "love", "tfidf"))

t0 = time.perf_counter()
for u in upiti[:100]:
    model.rangiraj(ind, u, "kosinus", fam)
print(f"-- 100 rangiranih upita: {(time.perf_counter() - t0) * 10:.2f} ms po upitu")
t0 = time.perf_counter()
model.slicne(ind, d, 10, fam)
print(f"-- slicne pesme za jednu pesmu: {(time.perf_counter() - t0) * 1000:.2f} ms")