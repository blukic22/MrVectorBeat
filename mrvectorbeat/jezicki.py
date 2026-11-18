# Jezicki model sa Dirichlet ublazavanjem: "Koja je ovo pesma?"
#
# Pesma d je jezicki model; ocena je log verovatnoce da je upit q "izgovoren"
# iz te pesme:
#   P(t|d) = (tf(t,d) + mu * P(t|C)) / (dl + mu),   P(t|C) = cf(t) / |C|
#   ocena(q, d) = zbir po terminima upita w_t * log P(t|d)
# cf(t) - koliko se puta termin javlja u celom katalogu, |C| - ukupan broj
# termina u katalogu, dl - broj termina pesme, w_t - tezina termina u upitu
# (1 za svako otkucano pojavljivanje, deo od ALFA za dodati sinonim).
#
# Zasto ublazavanje: bez njega je P(t|d) = 0 za rec koje nema u pesmi, pa je
# ocena -beskonacno i pesma kojoj fali jedna rec ispada. Sa ublazavanjem rec
# koja fali dobija malu verovatnocu iz celog kataloga, pa pesma koja sadrzi
# vecinu reci ostaje visoko.
# Zasto Dirichlet (a ne Jelinek-Mercer): stihovi imaju vrlo razlicite duzine.
# Kod Dirichlet-a je udeo kataloga mu / (dl + mu): kratka pesma (malo dokaza)
# se ublazava jako, duga slabo. JM bi svaku pesmu ublazio istim lambda.
#
# Efikasnost: ocena se rastavlja na
#   zbir_t w_t * log(mu * P(t|C))          - isti za sve pesme
#   - W * log(dl + mu)                     - zavisi samo od duzine pesme
#   + zbir_{t u d} w_t * log(1 + tf / (mu * P(t|C)))  - samo reci koje pesma ima
# (W = zbir tezina). Zato se kroz invertovani indeks obilaze samo pesme sa bar
# jednom reci upita, a pesme bez nijedne reci nemaju dokaz i ne ulaze.
# cf, |C|, duzine, log(dl + mu) i log(1 + tf / (mu P(t|C))) uz postings za
# izabrano mu racunaju se jednom, pri gradnji, i cuvaju u artefaktu.
#
# MU je izabran sa grafika na strani Jezicki model (MRR nad upitima za jednu
# pesmu iz qrels, bez i sa greskom), ne iz literature. MRR je najveci za malo
# mu (5-25: 0.98 bez greske, 0.97 sa greskom) i pada za veliko (1000: 0.88 i
# 0.77): prosecna pesma ima 218 termina, pa je kod mu = 1000 oko 80%
# verovatnoce iz kataloga i pesma gubi svoje reci. Ali za vrlo malo mu svaka
# rec koja fali spusta ocenu toliko da pobedjuje pesma sa vise RAZLICITIH reci
# upita, makar jednom, a ponavljanje (refren) se skoro ne racuna. Primer iz
# zadatka "found love in a place without hope": Rihanna je 7. za mu = 5,
# 4. za mu = 10, 1. od mu = 100. Izabrano je mu = 100: MRR 0.976 (bez greske)
# i 0.949 (sa greskom), malo ispod vrha, a radi za zapamcen deo refrena.
#
# Prosirenje: ako najbolja pesma ne sadrzi sve reci upita, reci koje joj
# NEDOSTAJU (verovatno zamenjene) dobijaju do SINONIMA_PO_RECI WordNet sinonima
# koji postoje u katalogu. Sinonimi jedne reci ZAJEDNO nose ALFA = 0.3 (svaki
# ALFA / broj sinonima), a otkucana rec 1, pa dodate reci uvek nose manje.

import math
from array import array
from collections import Counter
from functools import lru_cache

import nltk

from mrvectorbeat import tekst
from mrvectorbeat.rang import uredi

MU = 100.0
ALFA = 0.3                  # ukupna tezina sinonima jedne reci; otkucana rec ima 1
SINONIMA_PO_RECI = 3


class JezickiModel :
    def __init__(self, ind, mu : float = MU) :
        n = ind.n
        self.n, self.mu = n, mu
        duzine = [0] * n
        self.cf = {}
        for t, (pesme, odmak, _) in ind.recnik.items() :
            self.cf[t] = odmak[-1]                     # = zbir tf po svim pesmama
            for i, d in enumerate(pesme) :
                duzine[d] += odmak[i + 1] - odmak[i]
        self.duzine = array("I", duzine)
        self.ukupno = sum(duzine)                      # |C|
        self.pc = {t : c / self.ukupno for t, c in self.cf.items()}
        # delovi ocene za izabrano mu, da se pri upitu samo citaju
        self.log_duzine = array("d", (math.log(x + mu) for x in duzine))
        self.dokaz = {}                                # termin -> array('d') uz postings
        for t, (pesme, odmak, _) in ind.recnik.items() :
            mpc = mu * self.pc[t]
            self.dokaz[t] = array("d", (math.log(1 + (odmak[i + 1] - odmak[i]) / mpc)
                                        for i in range(len(pesme))))

    def ocene(self, ind, tezine : dict, mu : float = None) -> dict :
        """tezine: termin -> w_t. Vraca recnik pesma -> log P(q|d), samo za pesme
        sa bar jednim terminom. Termin kog nema u katalogu (P(t|C) = 0) se
        preskace. mu se moze zadati (za grafik); inace vazi izabrano."""
        tezine = {t : w for t, w in tezine.items() if t in self.pc and w > 0}
        if not tezine :
            return {}
        izabrano = mu is None or mu == self.mu
        mu = self.mu if mu is None else mu
        zbir = {}
        for t, w in tezine.items() :
            pesme, odmak, _ = ind.recnik[t]
            if izabrano :
                for d, x in zip(pesme, self.dokaz[t]) :
                    zbir[d] = zbir.get(d, 0.0) + w * x
            else :
                mpc = mu * self.pc[t]
                for i, d in enumerate(pesme) :
                    zbir[d] = zbir.get(d, 0.0) + w * math.log(1 + (odmak[i + 1] - odmak[i]) / mpc)
        osnova = sum(w * math.log(mu * self.pc[t]) for t, w in tezine.items())   # isto za sve pesme
        W = sum(tezine.values())
        if izabrano :
            return {d : osnova - W * self.log_duzine[d] + x for d, x in zbir.items()}
        return {d : osnova - W * math.log(self.duzine[d] + mu) + x for d, x in zbir.items()}


# --- upit: reci, tezine, prosirenje ------------------------------------------

def reci_upita(upit : str) :
    """Lista (rec kako je otkucana, koren) bez stop-reci, redom."""
    r = tekst.obradi(upit)
    return list(zip(r["bez_stop"], r["koreni"]))


def _wordnet() :
    from nltk.corpus import wordnet as wn
    try :
        wn.ensure_loaded()
    except LookupError :
        nltk.download("wordnet", quiet = True)
        wn.ensure_loaded()
    return wn


@lru_cache(maxsize = 4096)
def sinonimi(rec : str) -> tuple :
    """WordNet sinonimi reci, redom kojim ih WordNet daje (najcesci smisao prvi).
    Izostavljaju se izrazi od vise reci (go_for) i sama rec. Vraca tuple reci."""
    rez = []
    for s in _wordnet().synsets(rec) :
        for lema in s.lemma_names() :
            lema = lema.lower()
            if "_" in lema or "-" in lema or lema == rec or lema in rez :
                continue
            rez.append(lema)
    return tuple(rez)


def prosiri(reci, model, nedostaju) :
    """Za svaku otkucanu rec ciji koren je u nedostaju: do SINONIMA_PO_RECI
    sinonima koji postoje u katalogu i ciji koren nije vec u upitu.
    Vraca listu (sinonim, koren, tezina); sinonimi jedne reci zajedno nose ALFA."""
    u_upitu = {k for _, k in reci}
    dodate = []
    for rec, k in reci :
        if k not in nedostaju :
            continue
        nadjeni = []
        for s in sinonimi(rec) :
            koreni = tekst.termini(s)
            if len(koreni) != 1 or koreni[0] not in model.pc or koreni[0] in u_upitu :
                continue
            u_upitu.add(koreni[0])
            nadjeni.append((s, koreni[0]))
            if len(nadjeni) == SINONIMA_PO_RECI :
                break
        dodate += [(s, kor, ALFA / len(nadjeni)) for s, kor in nadjeni]
    return dodate


def pretraga(ind, model, upit : str, porodice = None, prosirenje : bool = True, mu : float = None) -> dict :
    """Rangira pesme po log P(q|d). Ako najbolja pesma ne sadrzi sve reci upita
    (rezultat "nije dovoljno dobar"), reci koje joj nedostaju se prosiruju
    sinonimima (manja tezina) i pretrazuje se ponovo.
    Vraca dict: reci [(rec, koren)], dodate [(sinonim, koren, tezina)],
    prosireno, rezultat [(pesma, ocena)], ocenjeno."""
    reci = reci_upita(upit)
    tezine = dict(Counter(k for _, k in reci))           # otkucana rec: 1 po pojavljivanju
    rezultat, ocenjeno = uredi(model.ocene(ind, tezine, mu), porodice)
    dodate = []
    if prosirenje and reci :
        nedostaju = _nedostaju(ind, rezultat, tezine)
        if nedostaju :
            dodate = prosiri(reci, model, nedostaju)
        if dodate :
            for _, k, w in dodate :
                tezine[k] = w
            rezultat, ocenjeno = uredi(model.ocene(ind, tezine, mu), porodice)
    return {"reci" : reci, "dodate" : dodate, "prosireno" : bool(dodate),
            "rezultat" : rezultat, "ocenjeno" : ocenjeno}


def _nedostaju(ind, rezultat, tezine) -> list :
    """Termini upita kojih nema u najboljoj pesmi (svi, ako rezultata nema)."""
    if not rezultat :
        return list(tezine)
    d = rezultat[0][0]
    return [t for t in tezine if not len(ind._pozicije_u(t, [d])[0])]


def pokrivenost(ind, termini, pesme) -> dict :
    """pesme: rastuca lista id-jeva. Vraca pesma -> {termin: da li ga pesma sadrzi}.
    Za svaki termin jedan prolaz sa dva pokazivaca (ind._pozicije_u)."""
    rez = {d : {} for d in pesme}
    for t in dict.fromkeys(termini) :
        for d, p in zip(pesme, ind._pozicije_u(t, pesme)) :
            rez[d][t] = bool(len(p))
    return rez
