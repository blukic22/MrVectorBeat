# Mere kvaliteta rang-liste za jedan upit.
#
# Svaka mera prima:
#   lista - rang-lista kako je korisnik vidi: kljucevi porodica, redom, svaka
#           porodica najvise jednom (rang.uredi zadrzava jednu verziju)
#   ocene - recnik porodica -> ocena iz qrels (1 ili 2); porodica koje nema
#           ima ocenu 0. Ocena iz fajla vazi za celu porodicu, jer korisnik
#           vidi samo jednu verziju pesme.
#   prag  - od koje ocene je pesma pogodak: 2 za upite za jednu pesmu (ocena 1
#           je druga pesma istog izvodjaca i nije pogodak), 1 za tematske.
# Ako upit nema nijednu relevantnu pesmu, mera vraca 0.

import math

GRUPE = ("Jedna pesma, bez greške", "Jedna pesma, sa greškom", "Tematski upiti")
GLAVNA = {GRUPE[0] : "MRR", GRUPE[1] : "MRR", GRUPE[2] : "nDCG@5"}
MERE = ("P@5", "MRR", "MAP", "nDCG@5")      # proseci preko upita jedne grupe


def grupa(skup : str, tipfeler : int) -> str :
    if skup == "tema" :
        return GRUPE[2]
    return GRUPE[1] if tipfeler else GRUPE[0]


def prag(skup : str) -> int :
    return 1 if skup == "tema" else 2


def p_na_5(lista, ocene : dict, prag : int) -> float :
    """Udeo pogodaka medju prvih 5. Deli se sa 5 i kad lista ima manje od 5."""
    if not any(o >= prag for o in ocene.values()) :
        return 0.0
    return sum(1 for f in lista[:5] if ocene.get(f, 0) >= prag) / 5


def reciprocni_rang(lista, ocene : dict, prag : int) -> float :
    """1 / mesto prvog pogotka (mesta od 1); 0 ako pogotka nema u listi."""
    for i, f in enumerate(lista, 1) :
        if ocene.get(f, 0) >= prag :
            return 1.0 / i
    return 0.0


def prosecna_preciznost(lista, ocene : dict, prag : int) -> float :
    """Prosek preciznosti na mestu svakog pogotka. Deli se brojem SVIH
    relevantnih porodica (i onih koje lista nije vratila): relevantna pesma
    koju model ne nadje doprinosi 0, ali ulazi u imenilac."""
    relevantnih = sum(1 for o in ocene.values() if o >= prag)
    if relevantnih == 0 :
        return 0.0
    pogodaka, zbir = 0, 0.0
    for i, f in enumerate(lista, 1) :
        if ocene.get(f, 0) >= prag :
            pogodaka += 1
            zbir += pogodaka / i
    return zbir / relevantnih


def ndcg_na_5(lista, ocene : dict, k : int = 5) -> float :
    """DCG@k = zbir rel_i / log2(i + 1), rel_i je ocena (2, 1 ili 0), pa ocena 2
    vredi dvostruko vise od ocene 1. Deli se idealnim DCG-om: najbolje ocene
    poredjane na prvih k mesta. Koristi ocene 2 i 1 u svim grupama."""
    idealne = sorted(ocene.values(), reverse = True)[:k]
    idcg = sum(o / math.log2(i + 1) for i, o in enumerate(idealne, 1))
    if idcg == 0 :
        return 0.0
    dcg = sum(ocene.get(f, 0) / math.log2(i + 1) for i, f in enumerate(lista[:k], 1))
    return dcg / idcg


def mere_upita(lista, ocene : dict, prag : int) -> dict :
    """Sve cetiri mere za jedan upit."""
    return {"P@5" : p_na_5(lista, ocene, prag),
            "MRR" : reciprocni_rang(lista, ocene, prag),
            "MAP" : prosecna_preciznost(lista, ocene, prag),
            "nDCG@5" : ndcg_na_5(lista, ocene)}
