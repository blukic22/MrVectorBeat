import math
from array import array
from collections import Counter

from mrvectorbeat import tekst

MERE = ("tfidf", "kosinus")


class TfidfModel :
    def __init__(self, ind) :
        n = ind.n
        self.n = n
        self.termini = list(ind.recnik)                      
        self.idf = {}
        kv_lnc = [0.0] * n
        kv_ltc = [0.0] * n
        for t in self.termini :
            pesme, odmak, _ = ind.recnik[t]
            idf = math.log10(n / len(pesme))
            self.idf[t] = idf
            for i, d in enumerate(pesme) :
                w = 1.0 + math.log10(odmak[i + 1] - odmak[i])
                kv_lnc[d] += w * w
                kv_ltc[d] += (w * idf) ** 2
        self.norma_lnc = array("d", (math.sqrt(x) for x in kv_lnc))
        self.norma_ltc = array("d", (math.sqrt(x) for x in kv_ltc))

        self.tezine = {}                                      
        pt = [array("I") for _ in range(n)]
        pf = [array("I") for _ in range(n)]
        for tid, t in enumerate(self.termini) :
            pesme, odmak, _ = ind.recnik[t]
            idf = self.idf[t]
            a, b, c = array("d"), array("d"), array("d")
            for i, d in enumerate(pesme) :
                tf = odmak[i + 1] - odmak[i]
                w = 1.0 + math.log10(tf)
                a.append(w * idf)
                b.append(w / self.norma_lnc[d])
                c.append(w * idf / self.norma_ltc[d] if self.norma_ltc[d] > 0 else 0.0)
                pt[d].append(tid)
                pf[d].append(tf)
            self.tezine[t] = (a, b, c)
        self.p_termini, self.p_tf = pt, pf

    def broj_tezina(self) :
        return sum(len(v[0]) for v in self.tezine.values())


    def ocene(self, ind, upit : str, mera : str) -> dict :
        if mera not in MERE :
            raise ValueError(f"Nepoznata mera: {mera}")
        tf_upita = Counter(tekst.termini(upit))
        ocene = {}
        if mera == "tfidf" :
            for t in tf_upita :                              
                tz = self.tezine.get(t)
                if tz is None :
                    continue
                for d, x in zip(ind.postings(t), tz[0]) :
                    ocene[d] = ocene.get(d, 0.0) + x
            return ocene
        wq = {t : (1.0 + math.log10(tf)) * self.idf[t] for t, tf in tf_upita.items() if t in self.idf}
        norma = math.sqrt(sum(w * w for w in wq.values()))
        for t, w in wq.items() :
            f = w / norma if norma > 0 else 0.0
            for d, x in zip(ind.postings(t), self.tezine[t][1]) :
                ocene[d] = ocene.get(d, 0.0) + f * x
        return ocene

    def rangiraj(self, ind, upit : str, mera : str, porodice = None) :
        ocene = self.ocene(ind, upit, mera)
        redosled = sorted(ocene.items(), key = lambda x : (-x[1], x[0]))
        if porodice is None :
            return redosled, len(ocene)
        videne, rez = {}, []
        for d, o in redosled :
            f = porodice[d]
            if f in videne :
                continue
            videne[f] = True
            rez.append((d, o))
        return rez, len(ocene)


    def slicne(self, ind, d : int, k : int = 10, porodice = None) -> list :
        norma = self.norma_ltc[d]
        if norma == 0 :
            return []
        ocene = {}
        for tid, tf in zip(self.p_termini[d], self.p_tf[d]) :
            t = self.termini[tid]
            w = (1.0 + math.log10(tf)) * self.idf[t] / norma
            if w == 0 :
                continue
            for p, x in zip(ind.postings(t), self.tezine[t][2]) :
                ocene[p] = ocene.get(p, 0.0) + w * x
        ocene.pop(d, None)
        redosled = sorted(ocene.items(), key = lambda x : (-x[1], x[0]))
        videne, rez = {}, []
        for p, o in redosled :
            if porodice is not None :
                f = porodice[p]
                if f == porodice[d] or f in videne :
                    continue
                videne[f] = True
            rez.append((p, o))
            if len(rez) == k :
                break
        return rez


    def doprinosi(self, a : int, b : int, n : int = 10) :
        na, nb = self.norma_ltc[a], self.norma_ltc[b]
        if na == 0 or nb == 0 :
            return 0.0, []
        A, FA, B, FB = self.p_termini[a], self.p_tf[a], self.p_termini[b], self.p_tf[b]
        i = j = 0
        ukupno, zajednicki = 0.0, []
        while i < len(A) and j < len(B) :
            if A[i] == B[j] :
                t = self.termini[A[i]]
                idf = self.idf[t]
                dop = ((1.0 + math.log10(FA[i])) * idf / na) * ((1.0 + math.log10(FB[j])) * idf / nb)
                ukupno += dop
                zajednicki.append((t, FA[i], FB[j], dop))
                i += 1
                j += 1
            elif A[i] < B[j] :
                i += 1
            else :
                j += 1
        zajednicki.sort(key = lambda x : -x[3])
        return ukupno, [(t, fa, fb, dop, dop / ukupno if ukupno > 0 else 0.0)
                        for t, fa, fb, dop in zajednicki[:n]]