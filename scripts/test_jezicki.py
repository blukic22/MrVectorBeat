# Provere za Nedelju 6 (jezicki model, prosirenje upita, plejlista).
# Pokretanje iz korena projekta:  python -m scripts.test_jezicki
# Nezavisan racun ide kroz SVE pesme iz tokeni (set sme samo u testu).

import math, random
from collections import Counter

from mrvectorbeat import pipeline, tekst, jezicki, plejlista

df = pipeline.ucitaj_katalog()
tok = pipeline.ucitaj_tokene()
ind = pipeline.ucitaj_indeks()
jm = pipeline.ucitaj_jezicki()
tfidf = pipeline.ucitaj_tfidf()
fam = df["porodica"].to_numpy()
ids = list(df["track_id"])
N = len(ids)
TF = [Counter(tok[t]) for t in ids]
DL = [len(tok[t]) for t in ids]
CF = Counter(t for t_id in ids for t in tok[t_id])
C = sum(DL)


def provera(opis, uslov):
    print(f"{'OK ' if uslov else 'GRESKA'} {opis}")


def brute(tezine, mu):
    tezine = {t: w for t, w in tezine.items() if t in CF}
    return {d: sum(w * math.log((TF[d][t] + mu * CF[t] / C) / (DL[d] + mu)) for t, w in tezine.items())
            for d in range(N) if any(t in TF[d] for t in tezine)}


def isto(x, y, tol=1e-7):
    return x.keys() == y.keys() and all(abs(x[d] - y[d]) < tol for d in x)


print("Jezicki model:")
provera(f"|C| = {C}, cf i duzine == nezavisan racun",
        jm.ukupno == C and jm.cf == dict(CF) and list(jm.duzine) == DL)
random.seed(6)
recnik = list(ind.recnik)
upiti = [dict(Counter(random.sample(recnik, random.randint(1, 6)))) for _ in range(30)]
upiti += [dict(Counter(tekst.termini("found love in a place without hope"))), {"love": 2, "heart": 1}]
provera(f"ocena == zbir w * log((tf + mu P(t|C)) / (dl + mu)) kroz svih {N} pesama "
        f"({len(upiti)} upita, mu = {jm.mu:g})", all(isto(jm.ocene(ind, q), brute(q, jm.mu)) for q in upiti))
provera("isto i za zadato mu (10 i 2000), bez unapred izracunatih delova",
        all(isto(jm.ocene(ind, q, mu), brute(q, mu)) for q in upiti[:8] for mu in (10, 2000)))
provera("ponderisan upit (sinonim tezine 0.1) == nezavisan racun",
        isto(jm.ocene(ind, {"love": 1, "heart": 0.1}), brute({"love": 1, "heart": 0.1}, jm.mu)))
provera(f"ocenjeno samo kandidata: {len(jm.ocene(ind, {'love': 1, 'heart': 1}))} od {N} (love heart)",
        len(jm.ocene(ind, {"love": 1, "heart": 1})) < N)

ok = True
for d in random.sample(range(N), 20):
    if len(TF[d]) < 4:
        continue
    imam = random.sample(sorted(TF[d]), 3)
    fali = next(t for t in recnik if t not in TF[d] and len(ind.postings(t)) > 50)
    rez = jezicki.uredi(jm.ocene(ind, {t: 1 for t in imam + [fali]}), None)[0]
    ok &= d in dict(rez)
provera("pesma kojoj fali jedna rec upita ostaje u rezultatu (20 pesama)", ok)

rih = df.index[(df["track_name"] == "We Found Love") & (df["track_artist"] == "Rihanna")].tolist()
r = jezicki.pretraga(ind, jm, "found love in a place without hope", fam)
prva = r["rezultat"][0][0]
provera(f"primer iz zadatka: prva je {df['track_name'].iat[prva]} - {df['track_artist'].iat[prva]}",
        fam[prva] == fam[rih[0]])
pok = jezicki.pokrivenost(ind, [k for _, k in r["reci"]], [prva])[prva]
provera("za nju nedostaju tacno without i hope",
        [rec for rec, k in r["reci"] if not pok[k]] == ["without", "hope"])

print("Prosirenje upita:")
provera(f"prosireno: {[s for s, _, _ in r['dodate']]}", r["prosireno"])
provera("svaka dodata rec ima tezinu < 1 (otkucana ima 1)", all(w < 1 for _, _, w in r["dodate"]))
provera(f"sinonimi jedne reci (hope) zajedno nose ALFA = {jezicki.ALFA}, manje od same reci",
        abs(sum(w for s, _, w in r["dodate"] if s in jezicki.sinonimi("hope")) - jezicki.ALFA) < 1e-12)
provera("sinonimi su samo za reci koje nedostaju (hope), ne za found/love/place",
        all(s in jezicki.sinonimi("hope") for s, _, _ in r["dodate"]))
provera("dodate reci postoje u katalogu i nisu vec u upitu",
        all(k in jm.pc and k not in {kk for _, kk in r["reci"]} for _, k, _ in r["dodate"]))
provera("prosirenje=False ne dodaje reci",
        jezicki.pretraga(ind, jm, "found love in a place without hope", fam, False)["dodate"] == [])
celo = jezicki.pretraga(ind, jm, "we found love in a hopeless place", fam)
provera("kad najbolja pesma ima sve reci, upit se ne prosiruje", not celo["prosireno"])
for u in ("", "the and of", "qqqzzz"):
    provera(f"upit {u!r}: prazan rezultat, bez pada", jezicki.pretraga(ind, jm, u, fam)["rezultat"] == [])
provera("nepostojeca rec se preskace: ocene(love, qqqzzz) == ocene(love)",
        jm.ocene(ind, {"love": 1, "qqqzzz": 1}) == jm.ocene(ind, {"love": 1}))

print("Plejlista:")
v = plejlista.vektor(tfidf, prva)
provera(f"vektor pesme je redak: {len(v)} termina (koliko pesma ima), ne {len(recnik)}",
        len(v) == len(TF[prva]) and abs(sum(w * w for w in v.values()) - 1) < 1e-9)
prikazane = [d for d, _ in r["rezultat"][:10]]
dopada, ne_dopada = prikazane[1:3], prikazane[3:4]
p = plejlista.napravi(ind, tfidf, dopada, ne_dopada, prikazane, fam)
iz_oznacenih = set().union(*(TF[d] for d in dopada))
provera(f"reci plejliste ({len(p['reci'])}) poticu iz pesama koje se dopadaju",
        all(t in iz_oznacenih for t, _ in p["reci"]))
provera("profil: rec samo iz pesme koja se ne dopada nema pozitivnu tezinu",
        all(t in iz_oznacenih for t in plejlista.profil(tfidf, dopada, ne_dopada)))
pl = [d for d, _ in p["plejlista"]]
provera(f"plejlista ima {len(pl)} pesama, jednu po porodici", len(pl) == 10 and len({fam[d] for d in pl}) == 10)
provera("u plejlisti nema prikazanih ni oznacenih pesama ni njihovih verzija",
        not ({fam[d] for d in pl} & {fam[d] for d in prikazane + dopada + ne_dopada}))
p2 = plejlista.napravi(ind, tfidf, [prikazane[2], prikazane[5]], ne_dopada, prikazane, fam)
provera("promena oznaka menja i reci i plejlistu",
        p2["reci"] != p["reci"] and p2["plejlista"] != p["plejlista"])
p3 = plejlista.napravi(ind, tfidf, prikazane[:3], [], prikazane, fam)
provera("bez oznaka (prva 3 rezultata) plejlista se pravi", len(p3["plejlista"]) == 10)
provera(f"svaka rec profila postoji u bar {plejlista.MIN_PESAMA} pesama",
        all(len(ind.postings(t)) >= plejlista.MIN_PESAMA for t, _ in p["reci"]))
