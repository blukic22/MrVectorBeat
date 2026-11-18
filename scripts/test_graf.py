# Provere za Nedelju 7 (crawler, PageRank, slicni izvodjaci).
# Pokretanje iz korena projekta:  python -m scripts.test_graf
# Mreza se ne koristi: crawler se proverava na laznoj sesiji i na kesu.

import math, random, time

import numpy as np

from mrvectorbeat import crawler, pagerank, pipeline


def provera(opis, uslov):
    print(f"{'OK ' if uslov else 'GRESKA'} {opis}")


def tacan_pagerank(linkovi, d):
    """Resenje sistema r = (1-d)/N + d * (M r + viseci/N) direktno (numpy, samo u testu)."""
    n = len(linkovi)
    M = np.zeros((n, n))
    for j, izl in enumerate(linkovi):
        if izl:
            for i in izl:
                M[i, j] += 1 / len(izl)
        else:
            M[:, j] = 1 / n
    return np.linalg.solve(np.eye(n) - d * M, np.full(n, (1 - d) / n))


print("PageRank:")
mali = [[1, 2], [2], [0], [2], []]               # 4 -> nista (bez izlaznih), 3 -> 2
r, it = pagerank.pagerank(mali, 0.85)
provera(f"mali graf == tacno resenje sistema ({it} iteracija)", np.allclose(r, tacan_pagerank(mali, 0.85), atol=1e-9))
provera("zbir rangova = 1 i sa stranicom bez izlaznih linkova", abs(sum(r) - 1) < 1e-9)
random.seed(7)
nasumican = [random.sample(range(300), random.randint(0, 8)) for _ in range(300)]
nasumican = [[i for i in l if i != j] for j, l in enumerate(nasumican)]
ok = all(np.allclose(pagerank.pagerank(nasumican, d)[0], tacan_pagerank(nasumican, d), atol=1e-8)
         for d in (0.5, 0.85, 0.95))
provera("nasumican graf od 300 stranica == tacno resenje za d = 0.5, 0.85, 0.95", ok)
provera("veci d -> vise iteracija", pagerank.pagerank(nasumican, 0.5)[1] < pagerank.pagerank(nasumican, 0.95)[1])
provera("dolaznih == broj stranica koje pokazuju", list(pagerank.dolaznih(mali)) == [1, 1, 3, 0, 0])
# X (0) i Y (1) imaju po 2 dolazna linka; na izvore od X (2, 3) pokazuje jos 10
# stranica, na izvore od Y (4, 5) niko
g2 = [[], [], [0], [0], [1], [1]] + [[2, 3] for _ in range(10)]
r2, _ = pagerank.pagerank(g2, 0.85)
provera(f"uticaj zavisi od vaznosti izvora: X i Y imaju po 2 dolazna, ali X {r2[0]:.4f} > Y {r2[1]:.4f}",
        pagerank.dolaznih(g2)[0] == pagerank.dolaznih(g2)[1] == 2 and r2[0] > r2[1])

print("Linkovi i naslovi:")
provera("normalizacija: _ -> razmak, %XX, #deo, prvo slovo",
        crawler.normalizuj("hip_hop_music#History") == "Hip hop music"
        and crawler.normalizuj("Guns_N%27_Roses") == "Guns N' Roses")
ocek = {"/wiki/Hip_hop_music": "Hip hop music", "https://en.wikipedia.org/wiki/Def_Jam_Recordings": "Def Jam Recordings",
        "./Rock_music": "Rock music", "/wiki/Category:Rappers": None, "/wiki/File:Logo.png": None,
        "/wiki/Talk:Queen_(band)": None, "/wiki/Template_talk:Infobox": None, "/wiki/Help:IPA": None,
        "https://fr.wikipedia.org/wiki/Queen": None, "https://example.com/x": None,
        "/w/index.php?title=Foo&action=edit&redlink=1": None, "/wiki/Foo?action=edit&redlink=1": None,
        "/wiki/Main_Page": None, "/wiki/Guitar_Hero:_Metallica": "Guitar Hero: Metallica"}
provera("filter linkova: kategorije, slike, razgovor, drugi jezici, crveni i spoljasnji otpadaju",
        all(crawler.naslov_iz_href(h) == t for h, t in ocek.items()))
html = ("<html><head><link rel=\"canonical\" href=\"https://en.wikipedia.org/wiki/Queen_(band)\"></head><body>"
        "<div id=\"mw-content-text\"><div class=\"mw-parser-output\">"
        "<div class=\"hatnote\">See <a href=\"/wiki/Queen_(disambiguation)\">x</a></div>"
        "<table class=\"infobox\"><tr><th class=\"infobox-label\">Genres</th><td><a href=\"/wiki/Rock_music\">Rock</a></td></tr>"
        "<tr><th class=\"infobox-label\">Years active</th><td>1970</td></tr></table>"
        "<p>The band album song <a href=\"/wiki/EMI_Records\">EMI</a> <a href=\"/wiki/Rock_music\">rock</a> "
        "<a href=\"/wiki/Category:Bands\">c</a> <a class=\"new\" href=\"/wiki/Nema\">n</a> "
        "<a href=\"/wiki/Queen_(band)\">sebe</a></p>"
        "<div class=\"navbox\"><a href=\"/wiki/Spisak_nagrada\">nav</a></div>"
        "<ol class=\"references\"><li><a href=\"/wiki/Referenca\">r</a></li></ol>"
        "</div></div><div id=\"catlinks\"></div></body></html>")
p = crawler.parsiraj(html)
provera(f"parser: linkovi iz teksta i infoboxa, bez navboxa, referenci, kategorija, crvenih i samog sebe: {p['linkovi']}",
        p["linkovi"] == ["Rock music", "EMI Records"])
provera("parser: kanonski naslov, napomena, prepoznat izvodjac i muzicka stranica",
        p["kanonski"] == "Queen (band)" and p["napomene"] == ["Queen (disambiguation)"] and p["izvodjac"] and p["muzicka"])

print("Pristojnost crawlera (lazna sesija, bez mreze):")


class Odgovor:
    def __init__(self, status=200, text="<html></html>"):
        self.status_code, self.text, self.headers = status, text, {}


class LaznaSesija:
    def __init__(self):
        self.vremena, self.headers = [], {}

    def get(self, url, timeout=30):
        self.vremena.append(time.monotonic())
        return Odgovor(200, html) if "robots" not in url else Odgovor(200, "User-agent: *\nDisallow: /w/\n")


import tempfile
from pathlib import Path
with tempfile.TemporaryDirectory() as tmp:
    c = crawler.Crawler.__new__(crawler.Crawler)
    c.kes, c.pauza, c.poslednji, c.zahteva, c.iz_kesa = Path(tmp), 1.0, 0.0, 0, 0
    c.sesija = LaznaSesija()
    import urllib.robotparser
    c.robots = urllib.robotparser.RobotFileParser()
    c.robots.parse(c._robots_txt().splitlines())
    for t in ["A", "B", "C"]:
        c.stranica(t)
    razmaci = [b - a for a, b in zip(c.sesija.vremena, c.sesija.vremena[1:])]
    provera(f"jedan po jedan zahtev, razmak >= 1 s: {[round(x, 2) for x in razmaci]}", all(x >= 0.999 for x in razmaci))
    n = c.zahteva
    c.stranica("A"); c.stranica("a")                 # isto, i malo prvo slovo = isti clanak
    provera("ponovno trazenje ide iz kesa (0 novih zahteva)", c.zahteva == n)
    provera("robots.txt: zabranjena adresa se ne trazi",
            not c.robots.can_fetch(crawler.USER_AGENT, crawler.OSNOVA + "/w/index.php?x=1"))
    provera("sopstveni User-Agent sa imenom bota i kontaktom",
            crawler.USER_AGENT.startswith("MrVectorBeatBot/") and "http" in crawler.USER_AGENT)
pravi = crawler.Crawler()
provera(f"pravi robots.txt (iz kesa) dozvoljava /wiki/, pauza {pravi.pauza} s >= 1",
        pravi.robots.can_fetch(crawler.USER_AGENT, crawler.url_clanka("Queen (band)")) and pravi.pauza >= 1)

print("Graf i slicnost (artefakti):")
g = pipeline.ucitaj_graf()
rang = pipeline.ucitaj_rangove()
provera(f"graf: {g['statistika']['stranica']} stranica, bez petlji i dvostrukih grana",
        all(len(set(l)) == len(l) and j not in l for j, l in enumerate(g["linkovi"])))
provera("naslovi su jedinstveni (jedan cvor po clanku)", len(set(g["naslovi"])) == len(g["naslovi"]))
provera("izvodjaci iz kataloga su oznaceni i imaju ime iz kataloga",
        len(g["izvodjac"]) > 0 and all(isinstance(v, list) and v for v in g["izvodjac"].values()))
provera(f"PageRank nad celim grafom: {len(rang['pr'])} rangova, zbir 1",
        len(rang["pr"]) == len(g["naslovi"]) and abs(sum(rang["pr"]) - 1) < 1e-6)
provera("izvodjaci poredjani po uticaju", all(rang["pr"][a] >= rang["pr"][b] for a, b in zip(rang["poredak"], rang["poredak"][1:])))
sl = rang["slicnost"]
A = len(sl.izvodjaci)
dol = {a: {p for p, l in enumerate(g["linkovi"]) if a in l} for a in sl.izvodjaci}
ka = {p: [a for a in l if a in g["izvodjac"]] for p, l in enumerate(g["linkovi"])}
w = {p: math.log(A / len(v)) for p, v in ka.items() if v}


def brute(a, b):
    na = math.sqrt(sum(w[p] ** 2 for p in dol[a]))
    nb = math.sqrt(sum(w[p] ** 2 for p in dol[b]))
    return sum(w[p] ** 2 for p in dol[a] & dol[b]) / (na * nb) if na and nb else 0.0


ok = ok_sim = ok_str = True
for a in random.sample(sl.izvodjaci, min(15, A)):
    for b, kos, stranice, ukupno in sl.slicni(a, 10):
        ok &= abs(kos - brute(a, b)) < 1e-9
        ok_sim &= abs(kos - dict((x, k) for x, k, _, _ in sl.slicni(b, A))[a]) < 1e-9
        ok_str &= all(a in g["linkovi"][p] and b in g["linkovi"][p] for p in stranice) and ukupno == len(dol[a] & dol[b])
provera("slicnost == nezavisan kosinus sa tezinom log(A / a_p)", ok)
provera("slicnost je simetricna", ok_sim)
provera("\"Oboje se pominju u\": svaka stranica pokazuje na oba izvodjaca", ok_str)
po_broju = sorted(sl.tezina, key=lambda p: len(sl.ka_izvodjacima[p]))
provera("stranica koja pokazuje na vise izvodjaca ima manju tezinu (log(A / a_p))",
        all(sl.tezina[p] >= sl.tezina[q] for p, q in zip(po_broju, po_broju[1:]))
        and all(abs(sl.tezina[p] - w[p]) < 1e-12 for p in w))
