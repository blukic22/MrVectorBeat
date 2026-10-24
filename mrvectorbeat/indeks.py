# Pozicioni invertovani indeks: jedna struktura za Bulovu, fraznu i
# blizinsku pretragu.
#
# Za svaki termin cuvaju se tri array('I'):
#   pesme    - rastuci id-jevi pesama u kojima se termin javlja
#   odmak    - odmak[i]..odmak[i+1] je deo niza pozicije koji pripada pesme[i]
#              (duzina len(pesme) + 1, poslednji je kraj niza)
#   pozicije - pozicije svih pesama redom, unutar pesme rastuce
# Bulova pretraga cita samo pesme, fraza i blizina i pozicije.
#
# Id pesme je redni broj reda u katalogu. Pesme se dodaju redom po id-ju, a
# pozicije unutar pesme rastu, pa su sve liste sortirane vec pri gradnji.
#
# Pozicija (odluka B): mesto termina u listi SVIH reci stiha, pre uklanjanja
# stop-reci. Stop-reci se ne indeksiraju, ali zauzimaju mesto, pa fraza i
# blizina mere rastojanje u stvarnim recima stiha.
#
# Svaka metoda prima tekst upita i obradjuje ga istim putem kao stihove.
# Upit koji se posle obrade svede na [] daje prazan rezultat.

from array import array

from mrvectorbeat import tekst
from mrvectorbeat.skupovi import presek, razlika, presek_vise, unija_vise


class Indeks :
    def __init__(self, tokeni, pozicije, track_ids) :
        """tokeni[id], pozicije[id]: termini pesme i njihove pozicije (paralelno),
        id = redni broj pesme u katalogu."""
        self.track_ids = list(track_ids)
        self.n = len(self.track_ids)
        self.recnik = {}        # termin -> (pesme, odmak, pozicije)
        for id_ in range(self.n) :
            for t, p in zip(tokeni[id_], pozicije[id_]) :
                ulaz = self.recnik.get(t)
                if ulaz is None :
                    ulaz = self.recnik[t] = (array("I"), array("I"), array("I"))
                pesme, odmak, poz = ulaz
                if not pesme or pesme[-1] != id_ :     # prvo pojavljivanje termina u ovoj pesmi
                    pesme.append(id_)
                    odmak.append(len(poz))
                poz.append(p)
        for pesme, odmak, poz in self.recnik.values() :
            odmak.append(len(poz))                     # kraj poslednje pesme

    def __len__(self) :
        return len(self.recnik)

    def broj_postings(self) :
        return sum(len(pesme) for pesme, _, _ in self.recnik.values())

    def broj_pozicija(self) :
        return sum(len(poz) for _, _, poz in self.recnik.values())

    def postings(self, t) :
        """Rastuca lista pesama sa terminom t; termin kog nema daje praznu listu."""
        ulaz = self.recnik.get(t)
        return ulaz[0] if ulaz else []

    def _pozicije_u(self, t, izabrane) -> list :
        """Pozicije termina t u svakoj od izabranih pesama (rastuca lista id-jeva);
        za pesmu bez termina prazna lista. Dva pokazivaca: i ide kroz pesme
        termina, jer su obe liste rastuce, pa se nijedna pesma ne trazi od pocetka."""
        ulaz = self.recnik.get(t)
        if ulaz is None :
            return [[] for _ in izabrane]
        pesme, odmak, poz = ulaz
        rez, i = [], 0
        for d in izabrane :
            while i < len(pesme) and pesme[i] < d :
                i += 1
            if i < len(pesme) and pesme[i] == d :
                rez.append(poz[odmak[i]:odmak[i + 1]])
            else :
                rez.append([])
        return rez

    # --- Bulova pretraga ---------------------------------------------------

    def bulov(self, op : str, upit : str, zabranjeni : str = "") -> list :
        """op: "I" (presek), "ILI" (unija), "NE".
        NE: razlika(osnova, unija(zabranjeni)); osnova je presek termina iz upit,
        a ako ih nema, sve pesme. Vraca rastucu listu id-jeva pesama."""
        termini = tekst.termini(upit)
        if op == "I" :
            return presek_vise([self.postings(t) for t in termini]) if termini else []
        if op == "ILI" :
            return unija_vise([self.postings(t) for t in termini])
        if op == "NE" :
            zabr = tekst.termini(zabranjeni)
            if not termini and not zabr :
                return []
            osnova = presek_vise([self.postings(t) for t in termini]) if termini else list(range(self.n))
            if not zabr :
                return osnova
            return razlika(osnova, unija_vise([self.postings(t) for t in zabr]))
        raise ValueError(f"Nepoznat operator: {op}")

    # --- Pozicioni upiti ---------------------------------------------------

    def frazni(self, upit : str) -> list :
        """Pesme u kojima termini fraze stoje na istim medjusobnim rastojanjima
        kao u upitu. Stop-reci na pocetku i kraju fraze otpadaju, a u sredini su
        dzoker (mesto mora postojati, rec moze biti bilo koja).
        Vraca listu (pesma, [pocetne pozicije], duzina_fraze)."""
        koreni, poz = tekst.termini_sa_pozicijama(upit)
        if not koreni :
            return []
        pomeraji = [p - poz[0] for p in poz]           # pomeraj od prvog termina
        duzina = pomeraji[-1] + 1
        kandidati = presek_vise([self.postings(t) for t in koreni])
        if not kandidati :
            return []
        po_terminu = [self._pozicije_u(t, kandidati) for t in koreni]
        rez = []
        for k, d in enumerate(kandidati) :
            # pocetak p je pogodak ako je termin i na p + pomeraj[i] za svako i;
            # pozicije termina i pomerimo za -pomeraj[i] i presecamo pocetke
            pocetci = list(po_terminu[0][k])
            for i in range(1, len(koreni)) :
                if not pocetci :
                    break
                m = pomeraji[i]
                pocetci = presek(pocetci, [p - m for p in po_terminu[i][k] if p >= m])
            if pocetci :
                rez.append((d, pocetci, duzina))
        return rez

    def blizina(self, rec1 : str, rec2 : str, k : int) -> list :
        """Pesme u kojima su dve reci na rastojanju abs(p - q) < k (bilo kojim
        redosledom). Svaka rec se mora svesti na tacno jedan termin, inace [].
        Ako su termini isti, par se racuna samo kao p < q.
        Vraca listu (pesma, [(p, q)]), p je pozicija prve, q druge reci."""
        t1, t2 = tekst.termini(rec1), tekst.termini(rec2)
        if len(t1) != 1 or len(t2) != 1 :
            return []
        a, b = t1[0], t2[0]
        kandidati = presek(self.postings(a), self.postings(b))
        if not kandidati :
            return []
        poz_a = self._pozicije_u(a, kandidati)
        poz_b = poz_a if a == b else self._pozicije_u(b, kandidati)
        rez = []
        for d, P, Q in zip(kandidati, poz_a, poz_b) :
            parovi = []
            if a == b :
                # P je rastuca: za svako p idemo udesno dok je rastojanje < k
                for i in range(len(P)) :
                    j = i + 1
                    while j < len(P) and P[j] - P[i] < k :
                        parovi.append((P[i], P[j]))
                        j += 1
            else :
                # prozor Q[lo:hi] su pozicije q sa p - k < q < p + k; kako p raste,
                # lo i hi se pomeraju samo napred
                lo = hi = 0
                for p in P :
                    while lo < len(Q) and Q[lo] <= p - k :
                        lo += 1
                    if hi < lo :
                        hi = lo
                    while hi < len(Q) and Q[hi] < p + k :
                        hi += 1
                    for q in Q[lo:hi] :
                        parovi.append((p, q))
            if parovi :
                rez.append((d, parovi))
        return rez
