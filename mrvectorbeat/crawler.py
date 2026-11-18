# Crawler za englesku Vikipediju: clanci o izvodjacima iz kataloga i sve na sta
# oni upucuju (zanrovi, izdavacke kuce, albumi, drugi izvodjaci).
#
# Pristojnost prema serveru:
#   - robots.txt se cita jednom (urllib.robotparser) i svaka adresa se proverava
#     pre zahteva; pauza je max(PAUZA, Crawl-delay iz robots.txt);
#   - jedan po jedan zahtev, izmedju dva zahteva bar PAUZA sekundi;
#   - sopstveni User-Agent (Wikimedia trazi ime bota i kontakt);
#   - svaka preuzeta stranica cuva se na disku (KES, gzip), pa ponovno
#     pokretanje ne salje iste zahteve; prekinut obilazak se samo nastavi.
#
# Linkovi: HTML parser (BeautifulSoup, "html.parser") samo nad tekstom clanka
# (div.mw-parser-output). Infobox ostaje (u njemu su zanr i izdavacka kuca), a
# izbacuju se navigacione kutije, sidebar, reference i napomene na vrhu, jer su
# to spiskovi koji ne govore o samom clanku. Prate se samo linkovi /wiki/Naslov
# bez imenskog prostora (Category:, File:, Talk:, Template: ...); druge jezicke
# verzije (fr.wikipedia.org) i spoljasnji linkovi nisu na en.wikipedia.org/wiki/,
# a crveni linkovi (stranica ne postoji, klasa "new", ?redlink=1) se preskacu.
# Isti clanak pod vise adresa: naslov se normalizuje (dekodiranje, _ -> razmak,
# bez #dela, prvo slovo veliko), a posle preuzimanja se koristi kanonski naslov
# iz <link rel="canonical">, pa preusmerenja padaju u isti cvor.
#
# Prednost muzici: ocena muzicnosti se racuna iz SADRZAJA stranice (muzicka
# polja infoboxa i udeo muzickih reci u tekstu). Prvo se obilaze clanci
# izvodjaca iz kataloga, pa linkovi po oceni muzicnosti stranice sa koje vode
# (prioritetni red, heapq).

import gzip
import hashlib
import heapq
import pickle
import re
import time
import urllib.robotparser
from pathlib import Path
from urllib.parse import quote, unquote

import requests
from bs4 import BeautifulSoup

OSNOVA = "https://en.wikipedia.org"
USER_AGENT = "MrVectorBeatBot/1.0 (studentski projekat; https://github.com/blukic22/MrVectorBeat)"
# van projekta (projekat je u OneDrive folderu, a kes ima oko 1-1.5 GB)
KES = Path.home() / "mrvectorbeat_cache" / "wiki"
PAUZA = 1.0
VERZIJA_PARSERA = 3          # promena parsera -> parsirani kes se pravi iznova (bez zahteva)

IMENSKI_PROSTORI = {
    "category", "file", "image", "media", "talk", "user", "wikipedia", "wp", "project",
    "help", "template", "portal", "special", "mediawiki", "module", "draft", "timedtext",
    "book", "gadget", "topic", "education program",
}
IZBACI = [                    # delovi clanka koji nisu tekst clanka
    ".navbox", ".navbox-styles", ".vertical-navbox", ".sidebar", "[role=navigation]",
    ".reflist", ".references", ".mw-references-wrap", ".mw-editsection", ".hatnote",
    ".ambox", ".metadata", ".sistersitebox", ".portalbox", ".navbar", "sup.reference",
]
POLJA_IZVODJACA = {"years active", "labels", "instruments", "members", "past members"}
POLJA_MUZIKE = {"genre", "genres", "label", "labels", "recorded", "released", "producer",
                "years active", "instruments", "length", "studio", "venue"}
MUZICKE_RECI = {
    "album", "albums", "single", "singles", "song", "songs", "band", "bands", "singer",
    "rapper", "musician", "music", "musical", "genre", "label", "records", "recording",
    "recorded", "discography", "billboard", "chart", "charts", "tour", "guitar", "guitarist",
    "drummer", "bassist", "vocals", "vocalist", "lyrics", "hip", "hop", "rock", "pop", "jazz",
    "rap", "dj", "producer", "ep", "remix", "concert", "grammy", "songwriter",
}
SUFIKSI_IZVODJACA = ("band", "musician", "rapper", "singer", "group", "dj", "duo",
                     "producer", "artist", "songwriter", "entertainer")
PRAG_MUZIKE = 2.0            # % muzickih reci u tekstu od kog je stranica muzicka


# --- naslovi i linkovi -------------------------------------------------------

def normalizuj(naslov : str) -> str :
    """Isti clanak -> isti naslov: dekodira %XX, _ -> razmak, bez #dela, visak
    razmaka, prvo slovo veliko (MediaWiki ne razlikuje velicinu prvog slova)."""
    naslov = unquote(naslov).split("#")[0].replace("_", " ")
    naslov = re.sub(r"\s+", " ", naslov).strip()
    return naslov[:1].upper() + naslov[1:]


PREFIKSI = ("https://en.wikipedia.org/wiki/", "//en.wikipedia.org/wiki/", "/wiki/", "./")


def naslov_iz_href(href : str) :
    """Naslov clanka iz href-a, ili None ako link ne vodi na clanak engleske
    Vikipedije. Vikipedija daje linkove u vise oblika (apsolutni
    https://en.wikipedia.org/wiki/X, /wiki/X, ./X); svi daju isti naslov."""
    if not href :
        return None
    for prefiks in PREFIKSI :
        if href.startswith(prefiks) :
            ostatak = href[len(prefiks):]
            break
    else :
        return None                                    # spoljasnji link ili drugi jezik (fr.wikipedia...)
    if "?" in ostatak :
        return None                                    # crveni link (?action=edit&redlink=1) i sl.
    naslov = normalizuj(ostatak)
    if not naslov or naslov == "Main Page" :
        return None
    if ":" in naslov :
        prostor = naslov.split(":", 1)[0].strip().lower()
        if prostor in IMENSKI_PROSTORI or prostor.endswith(" talk") :
            return None
    return naslov


def url_clanka(naslov : str) -> str :
    return OSNOVA + "/wiki/" + quote(naslov.replace(" ", "_"), safe = "()',!*:@&=+$;-._~")


def je_razdvajanje(html : str) -> bool :
    return ('id="disambigbox"' in html or "dmbox-disambig" in html
            or "Category:Disambiguation_pages" in html or "Category:All_disambiguation_pages" in html)


def parsiraj(html : str) -> dict :
    """Iz HTML-a clanka: kanonski naslov, linkovi (redom, bez ponavljanja),
    muzicka ocena, da li je clanak o izvodjacu, da li je stranica za
    razdvajanje i linkovi iz napomena na vrhu (za trazenje clanka izvodjaca)."""
    kraj_glave = html.find("</head>")
    glava = BeautifulSoup(html[:kraj_glave + 7] if kraj_glave > 0 else "", "html.parser")
    kan = glava.find("link", rel = "canonical")
    kanonski = None
    if kan and kan.get("href") :
        kanonski = naslov_iz_href("/wiki/" + kan["href"].split("/wiki/", 1)[-1])

    # samo telo clanka: od mw-content-text do kategorija (brze nego ceo dokument)
    pocetak = html.find('id="mw-content-text"')
    kraj = html.find('id="catlinks"', pocetak)
    telo = BeautifulSoup(html[pocetak - 5 : kraj if kraj > 0 else len(html)], "html.parser") \
        if pocetak > 0 else BeautifulSoup("", "html.parser")
    sadrzaj = telo.select_one(".mw-parser-output") or telo

    napomene = []                                      # "For the rapper, see Logic (rapper)"
    for h in sadrzaj.select(".hatnote a[href]") :
        t = naslov_iz_href(h["href"])
        if t :
            napomene.append(t)

    polja = {th.get_text(" ", strip = True).lower()
             for th in sadrzaj.select("table.infobox th.infobox-label")}
    for sel in IZBACI :
        for el in sadrzaj.select(sel) :
            el.decompose()

    linkovi, videni = [], {}
    for a in sadrzaj.find_all("a", href = True) :
        if "new" in (a.get("class") or []) :           # crveni link: clanak ne postoji
            continue
        t = naslov_iz_href(a["href"])
        if t and t not in videni and t != kanonski :
            videni[t] = True
            linkovi.append(t)

    reci = re.findall(r"[a-z]+", sadrzaj.get_text(" ").lower())
    udeo = 100 * sum(1 for r in reci if r in MUZICKE_RECI) / len(reci) if reci else 0.0
    # izvodjac: infobox ima zanr i bar jedno polje izvodjaca (album ima Genre i
    # Label u jednini, ali nema Years active / Labels / Members)
    izvodjac = ("genre" in polja or "genres" in polja) and len(polja & POLJA_IZVODJACA) >= 1
    muzicka = izvodjac or len(polja & POLJA_MUZIKE) >= 2 or udeo >= PRAG_MUZIKE
    return {"kanonski" : kanonski, "linkovi" : linkovi, "napomene" : napomene,
            "udeo_muzike" : udeo, "muzicka" : muzicka, "izvodjac" : izvodjac,
            "razdvajanje" : je_razdvajanje(html), "reci" : len(reci), "verzija" : VERZIJA_PARSERA}


# --- preuzimanje -------------------------------------------------------------

class Crawler :
    def __init__(self, kes : Path = KES, pauza : float = PAUZA) :
        self.kes = kes
        self.kes.mkdir(parents = True, exist_ok = True)
        self.sesija = requests.Session()
        self.sesija.headers["User-Agent"] = USER_AGENT
        self.poslednji = 0.0
        self.zahteva = 0                                # stvarno poslatih zahteva
        self.iz_kesa = 0
        self.pauza = pauza                              # dok se ne procita Crawl-delay
        self.robots = urllib.robotparser.RobotFileParser()
        self.robots.parse(self._robots_txt().splitlines())
        kasnjenje = self.robots.crawl_delay(USER_AGENT)
        self.pauza = max(pauza, float(kasnjenje) if kasnjenje else 0.0)

    def _robots_txt(self) -> str :
        putanja = self.kes / "robots.txt"
        if putanja.exists() :
            return putanja.read_text(encoding = "utf-8")
        odgovor = self._get(OSNOVA + "/robots.txt")    # nas User-Agent i za robots.txt
        putanja.write_text(odgovor.text, encoding = "utf-8")
        return odgovor.text

    def _get(self, url : str, pokusaja : int = 4) :
        """Jedan zahtev, bar self.pauza posle prethodnog. Na mreznu gresku ili
        429/503 (server preopterecen) ceka (Retry-After ili 10, 20, 40 s) i
        pokusava ponovo; posle poslednjeg pokusaja vraca None ili odgovor."""
        for i in range(pokusaja) :
            cekaj = self.poslednji + self.pauza - time.monotonic()
            if cekaj > 0 :
                time.sleep(cekaj)
            self.poslednji = time.monotonic()
            self.zahteva += 1
            try :
                odgovor = self.sesija.get(url, timeout = 30)
            except requests.RequestException :
                odgovor = None
            if odgovor is not None and odgovor.status_code not in (429, 503) :
                return odgovor
            if i + 1 < pokusaja :
                posle = odgovor.headers.get("Retry-After") if odgovor is not None else None
                time.sleep(int(posle) if posle and posle.isdigit() else 10 * 2 ** i)
        return odgovor

    def _putanja(self, naslov : str) -> Path :
        h = hashlib.sha1(naslov.encode("utf-8")).hexdigest()
        return self.kes / h[:2] / f"{h}.pkl.gz"

    def stranica(self, naslov : str) :
        """Parsirana stranica (dict iz parsiraj, plus status), iz kesa ako postoji.
        None ako robots.txt zabranjuje adresu."""
        naslov = normalizuj(naslov)
        putanja = self._putanja(naslov)
        if putanja.exists() :
            with gzip.open(putanja, "rb") as f :
                zapis = pickle.load(f)
            self.iz_kesa += 1
            if zapis["status"] == 200 and zapis["parsirano"].get("verzija") != VERZIJA_PARSERA :
                zapis["parsirano"] = parsiraj(zapis["html"])   # novi parser, isti HTML, bez zahteva
                self._snimi(putanja, zapis)
            return zapis
        url = url_clanka(naslov)
        if not self.robots.can_fetch(USER_AGENT, url) :
            return None
        odgovor = self._get(url)
        if odgovor is None :                           # mreza ne radi: bez kesiranja, probace se opet
            return None
        zapis = {"naslov" : naslov, "url" : url, "status" : odgovor.status_code,
                 "html" : odgovor.text if odgovor.status_code == 200 else "",
                 "parsirano" : None}
        if odgovor.status_code == 200 :
            zapis["parsirano"] = parsiraj(zapis["html"])
        if odgovor.status_code in (200, 404) :         # ostalo (npr. 429, 503) se ne kesira
            self._snimi(putanja, zapis)
        return zapis

    def _snimi(self, putanja, zapis) :
        putanja.parent.mkdir(exist_ok = True)
        with gzip.open(putanja, "wb") as f :
            pickle.dump(zapis, f, protocol = pickle.HIGHEST_PROTOCOL)


# --- obilazak ----------------------------------------------------------------

def nadji_clanak_izvodjaca(crawler, ime : str, pokusaja : int = 3) :
    """Clanak izvodjaca za ime iz kataloga: /wiki/Ime; ako to nije clanak o
    izvodjacu, kandidati su linkovi sa muzickim sufiksom ("Logic (rapper)") iz
    napomene na vrhu ili sa stranice za razdvajanje. Vraca zapis ili None."""
    zapis = crawler.stranica(ime)
    if zapis is None or zapis["status"] != 200 :
        return None
    p = zapis["parsirano"]
    if p["izvodjac"] and not p["razdvajanje"] :
        return zapis
    osnova = normalizuj(ime).lower()
    izvori = p["linkovi"] if p["razdvajanje"] else p["napomene"]
    kandidati = [t for t in izvori if t.lower().startswith(osnova + " (")
                 and any(s in t.lower() for s in SUFIKSI_IZVODJACA)]
    if not p["razdvajanje"] :                          # "Logic" -> "Logic (disambiguation)"
        kandidati += [t for t in p["napomene"] if t.lower() == osnova + " (disambiguation)"]
    for t in kandidati[:pokusaja] :
        z = crawler.stranica(t)
        if z is None or z["status"] != 200 :
            continue
        if z["parsirano"]["razdvajanje"] :
            for t2 in z["parsirano"]["linkovi"] :
                if t2.lower().startswith(osnova + " (") and any(s in t2.lower() for s in SUFIKSI_IZVODJACA) :
                    z2 = crawler.stranica(t2)
                    if z2 and z2["status"] == 200 and z2["parsirano"]["izvodjac"] :
                        return z2
                    break
        elif z["parsirano"]["izvodjac"] :
            return z
    return None


def obidji(imena, max_strana : int, crawler = None, ispis = print) -> dict :
    """imena: izvodjaci iz kataloga, redom po prednosti. Prvo se traze njihovi
    clanci, pa se obilaze linkovi po muzicnosti izvorne stranice, dok graf nema
    max_strana cvorova. Vraca graf (vidi napravi_graf)."""
    crawler = crawler or Crawler()
    cvorovi = {}                    # kanonski naslov -> parsirano
    alias = {}                      # trazeni naslov -> kanonski (preusmerenja)
    izvodjac_cvora = {}             # kanonski naslov -> [imena iz kataloga]
    t0 = time.time()

    def dodaj(zapis) :
        """Dodaje stranicu u graf; vraca (kanonski naslov ili None, da li je nova)."""
        p = zapis["parsirano"]
        k = p["kanonski"] or zapis["naslov"]
        alias[zapis["naslov"]] = k
        if p["razdvajanje"] :                          # stranica za razdvajanje nije clanak
            return None, False
        if k in cvorovi :                              # druga adresa istog clanka
            return k, False
        cvorovi[k] = p
        return k, True

    for i, ime in enumerate(imena, 1) :
        if len(cvorovi) >= max_strana :
            break
        z = nadji_clanak_izvodjaca(crawler, ime)
        if z is not None :
            k, _ = dodaj(z)
            if k :
                izvodjac_cvora.setdefault(k, []).append(ime)
        if i % 200 == 0 :
            ispis(f"  izvodjaci: {i}/{len(imena)}, nadjeno clanaka {len(izvodjac_cvora)}, "
                  f"zahteva {crawler.zahteva}, iz kesa {crawler.iz_kesa}, {time.time() - t0:.0f} s")

    red, broj = [], 0                                  # (-prioritet, redni broj, naslov)
    def gurni(izvor) :
        nonlocal broj
        prioritet = (2.0 if izvor["izvodjac"] else 1.0 if izvor["muzicka"] else 0.0) + izvor["udeo_muzike"] / 100
        for t in izvor["linkovi"] :
            if t not in alias and t not in cvorovi :
                broj += 1
                heapq.heappush(red, (-prioritet, broj, t))
    for p in list(cvorovi.values()) :
        gurni(p)
    while red and len(cvorovi) < max_strana :
        _, _, t = heapq.heappop(red)
        if t in alias or t in cvorovi :
            continue
        z = crawler.stranica(t)
        if z is None or z["status"] != 200 :
            alias[t] = None
            continue
        k, novo = dodaj(z)
        if novo :
            gurni(z["parsirano"])
        if novo and len(cvorovi) % 500 == 0 :
            ispis(f"  stranica u grafu: {len(cvorovi)}/{max_strana}, zahteva {crawler.zahteva}, "
                  f"iz kesa {crawler.iz_kesa}, {time.time() - t0:.0f} s")
    return napravi_graf(cvorovi, alias, izvodjac_cvora, crawler)


def napravi_graf(cvorovi, alias, izvodjac_cvora, crawler) -> dict :
    """Graf preuzetih stranica: cvor = clanak (kanonski naslov), grana = link ka
    drugom preuzetom clanku (preko alias-a, pa preusmerenja vode u isti cvor).
    Vise linkova ka istom clanku = jedna grana; link na sebe se izbacuje."""
    naslovi = list(cvorovi)
    id_ = {t : i for i, t in enumerate(naslovi)}
    linkovi = []
    for t in naslovi :
        izlazni, videni = [], {}
        for l in cvorovi[t]["linkovi"] :
            k = alias.get(l, l)
            j = id_.get(k) if k else None
            if j is not None and j != id_[t] and j not in videni :
                videni[j] = True
                izlazni.append(j)
        linkovi.append(izlazni)
    muzicka = [cvorovi[t]["muzicka"] for t in naslovi]
    return {
        "naslovi" : naslovi,
        "linkovi" : linkovi,                                   # id -> lista id-jeva
        "muzicka" : muzicka,
        "izvodjac" : {id_[t] : imena for t, imena in izvodjac_cvora.items()},   # cvor -> imena iz kataloga
        "statistika" : {"stranica" : len(naslovi), "muzickih" : sum(muzicka),
                        "grana" : sum(len(l) for l in linkovi), "zahteva" : crawler.zahteva,
                        "iz_kesa" : crawler.iz_kesa, "izvodjaca" : len(izvodjac_cvora)},
    }
