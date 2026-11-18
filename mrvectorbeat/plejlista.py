# "Plejlista po mom ukusu": Rocchio nad ltc vektorima pesama iz TF-IDF modela.
#
#   profil = BETA * prosek(vektori pesama koje se dopadaju)
#          - GAMA * prosek(vektori pesama koje se ne dopadaju)
#
# Vektor pesme je ltc: (1 + log10 tf) * idf / norma_ltc, samo za termine koje
# pesma ima (TfidfModel.p_termini / p_tf), pa nema tabele "sve pesme x sve
# reci": profil je recnik sa recima iz oznacenih pesama. Upit korisnika ne
# ulazi u profil (ALFA = 0), da bi reci plejliste poticale iz oznacenih pesama.
# Zadrzava se RECI termina sa najvecom pozitivnom tezinom (medju onima koji
# postoje u bar MIN_PESAMA pesama); one su "reci na
# osnovu kojih je plejlista napravljena". Tim recima se kroz invertovani indeks
# ocenjuju pesme (skalarni proizvod sa ltc tezinama uz postings), i bira se
# DUZINA pesama, jedna po porodici, bez pesama koje su vec prikazane i bez
# njihovih verzija.

import math
from collections import Counter

from mrvectorbeat import tekst
from mrvectorbeat.rang import uredi

BETA, GAMA = 0.75, 0.25
RECI = 15
DUZINA = 10
PODRAZUMEVANO = 3           # bez oznaka: pretpostavlja se da se dopadaju prva 3
MIN_PESAMA = 5              # rec iz manje pesama (npr. samo iz oznacene) ne nalazi nove pesme


def vektor(tfidf, d : int) -> dict :
    """ltc vektor pesme d: termin -> tezina (samo termini koje pesma ima)."""
    norma = tfidf.norma_ltc[d]
    if norma == 0 :
        return {}
    return {tfidf.termini[tid] : (1.0 + math.log10(tf)) * tfidf.idf[tfidf.termini[tid]] / norma
            for tid, tf in zip(tfidf.p_termini[d], tfidf.p_tf[d])}


def profil(tfidf, dopada, ne_dopada) -> dict :
    """Rocchio profil: termin -> tezina; zadrzavaju se samo pozitivne tezine."""
    p = {}
    for pesme, koef in ((dopada, BETA), (ne_dopada, -GAMA)) :
        if not pesme :
            continue
        for d in pesme :
            for t, w in vektor(tfidf, d).items() :
                p[t] = p.get(t, 0.0) + koef * w / len(pesme)
    return {t : w for t, w in p.items() if w > 0}


def napravi(ind, tfidf, dopada, ne_dopada, iskljuci, porodice) -> dict :
    """dopada / ne_dopada: liste id-jeva pesama; iskljuci: id-jevi pesama koje
    su vec prikazane. Iz plejliste se izbacuju te pesme, oznacene pesme i sve
    njihove verzije (porodice).
    Vraca dict: reci [(termin, tezina)], plejlista [(pesma, ocena)]."""
    p = profil(tfidf, dopada, ne_dopada)
    p = {t : w for t, w in p.items() if len(ind.postings(t)) >= MIN_PESAMA}
    reci = sorted(p.items(), key = lambda x : (-x[1], x[0]))[:RECI]
    ocene = {}
    for t, w in reci :
        for d, x in zip(ind.postings(t), tfidf.tezine[t][2]) :      # ltc tezine uz postings
            ocene[d] = ocene.get(d, 0.0) + w * x
    zabranjene = {porodice[d] for d in list(iskljuci) + list(dopada) + list(ne_dopada)}
    rez = [(d, o) for d, o in uredi(ocene, porodice)[0] if porodice[d] not in zabranjene][:DUZINA]
    return {"reci" : reci, "plejlista" : rez}


def oblici(df, pesme, termini) -> dict :
    """Za prikaz: koren -> najcesci otkucani oblik u datim pesmama (babi -> baby)."""
    trazeni = set(termini)
    brojac = {}
    for d in pesme :
        r = tekst.obradi(df["lyrics"].iat[d])
        for rec, k in zip(r["bez_stop"], r["koreni"]) :
            if k in trazeni :
                brojac.setdefault(k, Counter())[rec] += 1
    return {k : (brojac[k].most_common(1)[0][0] if k in brojac else k) for k in termini}
