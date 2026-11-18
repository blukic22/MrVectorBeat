# BM25 (Okapi): treci model rangiranja.
#
#   ocena(q, d) = zbir po RAZLICITIM terminima t iz upita od
#                 idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
#
#   tf    - broj pojavljivanja termina u pesmi (odmak[i+1] - odmak[i] iz indeksa)
#   dl    - duzina pesme = broj termina pesme posle obrade (bez stop-reci)
#   avgdl - prosecna duzina pesme u katalogu
#   k1    - koliko brzo tf "zasicuje": k1 = 0 znaci da je bitno samo da li se
#           termin javlja, veliko k1 znaci da tf raste skoro linearno
#   b     - koliko se kaznjava duga pesma: b = 0 bez normalizacije duzine,
#           b = 1 puna normalizacija
#   idf   - log10(1 + (N - df + 0.5) / (df + 0.5)). Zbog "1 +" argument
#           logaritma je uvek > 1, pa idf nikad nije negativan (u klasicnoj
#           formuli bez "1 +" termin u vise od pola pesama dobija idf < 0).
#
# idf, dl i avgdl ne zavise od upita: racunaju se jednom, pri gradnji, jednim
# prolazom kroz invertovani indeks, i cuvaju u artefaktu. Pri upitu se samo
# citaju; ocenjuju se samo pesme iz postings listi termina upita.
#
# K1 i B su izabrani sa grafika na strani Evaluacija (MRR za mrezu k1 x b nad
# upitima za jednu pesmu bez greske), ne iz literature. MRR je najveci za malo
# k1: upit je sest reci iz stiha, pa je bitnije koliko RAZLICITIH reci upita
# pesma sadrzi nego koliko se puta svaka ponavlja. Izabrano je k1 = 0.5, b = 0.5
# (MRR 0.990): sredina ravnog dela, susedne tacke su 0.975-0.990. Vrh mreze
# (k1 = 0.25, b = 0.75, MRR 0.993) je bolji za pola upita od 147, a pojedinacan
# vrh na istim upitima na kojima se meri lako je slucajan.

import math
from array import array

from mrvectorbeat import tekst

K1 = 0.5
B = 0.5


class Bm25Model :
    def __init__(self, ind, k1 : float = K1, b : float = B) :
        n = ind.n
        self.n = n
        self.k1, self.b = k1, b
        self.idf = {}
        duzine = [0] * n
        for t, (pesme, odmak, _) in ind.recnik.items() :
            df = len(pesme)
            self.idf[t] = math.log10(1.0 + (n - df + 0.5) / (df + 0.5))
            for i, d in enumerate(pesme) :
                duzine[d] += odmak[i + 1] - odmak[i]
        self.duzine = array("I", duzine)
        self.avgdl = sum(duzine) / n if n else 0.0
        # dl / avgdl za svaku pesmu, da se pri upitu ne deli iznova
        self.rel_duzina = array("d", (x / self.avgdl if self.avgdl > 0 else 0.0 for x in duzine))

    def ocene(self, ind, upit : str, mera : str = "bm25", k1 : float = None, b : float = None) -> dict :
        """Recnik pesma -> BM25 ocena, samo za pesme sa bar jednim terminom upita.
        k1 i b se mogu zadati (za grafik u evaluaciji); inace vaze izabrani."""
        if mera != "bm25" :
            raise ValueError(f"Nepoznata mera: {mera}")
        k1 = self.k1 if k1 is None else k1
        b = self.b if b is None else b
        ocene = {}
        for t in dict.fromkeys(tekst.termini(upit)) :     # razliciti termini, redom
            ulaz = ind.recnik.get(t)
            if ulaz is None :                              # termin kog nema se preskace
                continue
            pesme, odmak, _ = ulaz
            idf = self.idf[t]
            for i, d in enumerate(pesme) :
                tf = odmak[i + 1] - odmak[i]
                ocene[d] = ocene.get(d, 0.0) + idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * self.rel_duzina[d]))
        return ocene
