# PageRank i slicnost izvodjaca nad grafom stranica sa Vikipedije.
#
# PageRank (rucno, iterativno):
#   r_novo(i) = (1 - d) / N + d * ( zbir po j -> i od r(j) / izlaznih(j)
#                                    + (zbir r(j) po stranicama bez izlaznih) / N )
# Uticaj stranice zavisi od toga koliko je vaznih stranica pokazuje na nju, a
# ne samo od broja dolaznih linkova. d je faktor prigusenja: sa verovatnocom d
# "surfer" prati link, a sa 1 - d skace na nasumicnu stranicu (bez toga bi se
# zaglavio u stranicama bez izlaznih linkova ili zatvorenim grupama). Stranica
# bez izlaznih linkova svoj rang deli ravnomerno na sve stranice, pa je zbir
# rangova uvek 1. Iteracija staje kad je zbir |r_novo - r| manji od TOL.
#
# Slicnost izvodjaca ("Fans also like"): izvodjac je vektor stranica koje
# pokazuju na njega. Stranica p ima tezinu w(p) = log(A / a_p), gde je A broj
# izvodjaca iz kataloga u grafu, a a_p na koliko njih p pokazuje: spisak
# dobitnika nagrade (veliko a_p) vredi malo, clanak o malom zanru mnogo (kao
# idf). Slicnost je kosinus:  zbir po zajednickim p od w(p)^2 / (|a| * |b|).
# Kandidati su samo izvodjaci na koje pokazuju iste stranice (preko dolaznih
# linkova), bez tabele izvodjaci x izvodjaci.

import math
from array import array

from mrvectorbeat.skupovi import presek

D = 0.85
TOL = 1e-10


def pagerank(linkovi, d : float = D, tol : float = TOL, max_iter : int = 500) :
    """linkovi: id -> lista id-jeva (bez ponavljanja). Vraca (array rangova, broj iteracija)."""
    n = len(linkovi)
    if n == 0 :
        return array("d"), 0
    r = [1.0 / n] * n
    izlaznih = [len(l) for l in linkovi]
    for it in range(1, max_iter + 1) :
        novo = [0.0] * n
        viseci = 0.0                                   # rang stranica bez izlaznih linkova
        for j, izl in enumerate(linkovi) :
            if izlaznih[j] == 0 :
                viseci += r[j]
                continue
            deo = r[j] / izlaznih[j]
            for i in izl :
                novo[i] += deo
        osnova = (1 - d) / n + d * viseci / n
        novo = [osnova + d * x for x in novo]
        razlika = sum(abs(a - b) for a, b in zip(novo, r))
        r = novo
        if razlika < tol :
            break
    return array("d", r), it


def dolaznih(linkovi) -> array :
    """Broj stranica koje pokazuju na svaku stranicu."""
    broj = [0] * len(linkovi)
    for izl in linkovi :
        for i in izl :
            broj[i] += 1
    return array("I", broj)


class Slicnost :
    def __init__(self, graf) :
        izv = graf["izvodjac"]                         # cvor -> imena iz kataloga
        self.izvodjaci = sorted(izv)
        A = len(self.izvodjaci)
        je_izv = [False] * len(graf["naslovi"])
        for a in self.izvodjaci :
            je_izv[a] = True
        self.ka_izvodjacima = {}                       # stranica -> izvodjaci na koje pokazuje
        self.dolazni = {a : array("I") for a in self.izvodjaci}   # izvodjac -> stranice (rastuce)
        for p, izl in enumerate(graf["linkovi"]) :
            ka = sorted(a for a in izl if je_izv[a])
            if ka :
                self.ka_izvodjacima[p] = array("I", ka)
                for a in ka :
                    self.dolazni[a].append(p)          # p raste, pa su liste sortirane
        self.tezina = {p : math.log(A / len(ka)) for p, ka in self.ka_izvodjacima.items()}
        self.norma = {a : math.sqrt(sum(self.tezina[p] ** 2 for p in self.dolazni[a]))
                      for a in self.izvodjaci}

    def slicni(self, a : int, k : int = 10, n_stranica : int = 5) :
        """Najslicniji izvodjaci za a: lista (b, kosinus, [zajednicke stranice
        po tezini opadajuce, najvise n_stranica], broj zajednickih)."""
        na = self.norma.get(a, 0.0)
        if na == 0 :
            return []
        zbir = {}
        for p in self.dolazni[a] :
            w2 = self.tezina[p] ** 2
            if w2 == 0 :
                continue
            for b in self.ka_izvodjacima[p] :
                if b != a :
                    zbir[b] = zbir.get(b, 0.0) + w2
        ocene = sorted(((b, x / (na * self.norma[b])) for b, x in zbir.items()),
                       key = lambda x : (-x[1], x[0]))[:k]
        rez = []
        for b, kos in ocene :
            zajednicke = presek(self.dolazni[a], self.dolazni[b])
            zajednicke.sort(key = lambda p : (-self.tezina[p], p))
            rez.append((b, kos, zajednicke[:n_stranica], len(zajednicke)))
        return rez
