import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string


apostrofi = "'\u2019\u2018\u02bc`\u00b4"

def skloni_apostrofe(tekst : str) :
    for znak in apostrofi :
        tekst = "".join(tekst.split(znak))
    return tekst

def ucitaj_stop_reci() -> frozenset :
    try :
        words = stopwords.words("english")
    except LookupError :
        nltk.download("stopwords", quiet = True)
        words = stopwords.words("english")
    return frozenset(skloni_apostrofe(r.lower()) for r in words)

stop_reci = ucitaj_stop_reci()

def mala_slova(tekst) :
    return tekst.lower()

def izdvajanje_reci(tekst) -> list[str] :
    tekst_bez_apostrofa = skloni_apostrofe(tekst)
    return re.findall(r'[^\W_]+', tekst_bez_apostrofa)

def skloni_stop(tekst) :
    return [k for k in tekst if k not in stop_reci]

class Stemer :
    def __init__(self) :
        self.stemmer = SnowballStemmer("english")
        self.kes : dict[str,str] = {}
        self.poziva_stemer = 0

    def koren(self, rec : str) :
        k = self.kes.get(rec)
        if k is None :
            k = self.stemmer.stem(rec)
            self.kes[rec] = k
            self.poziva_stemer += 1
        return k

    def stemuj(self, reci : list[str]) -> list[str] :
        koren = self.koren
        return [koren(r) for r in reci]

stemer = Stemer()

def stemuj(words : list[str], stemer : Stemer = stemer) -> list[str] :
    return stemer.stemuj(words)

def obradi(tekst : str, stemer : Stemer = stemer) -> dict :
    mala = mala_slova(tekst)
    reci = izdvajanje_reci(mala)
    bez_stop = skloni_stop(reci)
    return {"mala_slova" :  mala, "reci" : reci, "bez_stop" : bez_stop, "koreni" : stemer.stemuj(bez_stop) }

def termini(tekst : str, stemer : Stemer = stemer) -> list[str] :
    return obradi(tekst, stemer)["koreni"]