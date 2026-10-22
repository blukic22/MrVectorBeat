import heapq


def oblik_obrasca(obrazac : str) -> str :
    """Koja struktura resava obrazac: "stablo", "obrnuto stablo" ili "k-gramski indeks"."""
    z = obrazac.count("*")
    if z == 0 or (z == 1 and obrazac.endswith("*")) :
        return "stablo"
    if z == 1 and obrazac.startswith("*") :
        return "obrnuto stablo"
    return "k-gramski indeks"


class Trie :
    """Prefiksno stablo nad imenima (mala slova). Cvorovi su brojevi (indeksi u listama),
    pa se stablo snima pickle-om bez duboke rekurzije.
    obrnuto=True -> imena se upisuju naopako, pa prefiks obrnutog imena = sufiks imena."""

    def __init__(self, obrnuto : bool = False) :
        self.obrnuto = obrnuto
        self.deca = [{}]      # deca[c] = {slovo: cvor deteta}
        self.maks = [0]       # maks[c] = najveca ucestanost imena u podstablu cvora c
        self.stavke = [[]]    # stavke[c] = imena koja se zavrsavaju u c: (broj, ime, tip)

    def __len__(self) :
        return len(self.deca)

    def dodaj(self, ime : str, tip : str, broj : int) :
        kljuc = ime.lower()
        if self.obrnuto :
            kljuc = kljuc[::-1]
        c = 0
        self.maks[c] = max(self.maks[c], broj)
        for slovo in kljuc :
            sledeci = self.deca[c].get(slovo)
            if sledeci is None :
                sledeci = len(self.deca)
                self.deca[c][slovo] = sledeci
                self.deca.append({})
                self.maks.append(0)
                self.stavke.append([])
            c = sledeci
            self.maks[c] = max(self.maks[c], broj)
        self.stavke[c].append((broj, ime, tip))

    def _cvor(self, prefiks : str) :
        c = 0
        for slovo in prefiks :
            c = self.deca[c].get(slovo)
            if c is None :
                return None
        return c

    def najcesca(self, prefiks : str, k : int = 10) :
        """k najcescih imena koja pocinju sa prefiks (kod obrnutog: zavrsavaju se sa prefiks).
        Vraca (lista (broj, ime, tip), broj posecenih cvorova)."""
        kljuc = prefiks.lower()
        if self.obrnuto :
            kljuc = kljuc[::-1]
        c = self._cvor(kljuc)
        if c is None :
            return [], 0
        # U redu su i cvorovi (prioritet = maks podstabla) i gotova imena (prioritet = broj).
        # Kad se izvadi ime, nijedan cvor u redu ne moze dati cesce ime (maks je gornja granica),
        # pa je ime sigurno sledece po redu. Staje se cim ima k imena.
        # Drugi element: 0 = ime, 1 = cvor, da pri jednakom prioritetu ime izadje prvo.
        red = [(-self.maks[c], 1, c, None)]
        rezultat, poseceno = [], 0
        while red and len(rezultat) < k :
            _, je_cvor, c, stavka = heapq.heappop(red)
            if not je_cvor :
                rezultat.append(stavka)
                continue
            poseceno += 1
            for s in self.stavke[c] :
                heapq.heappush(red, (-s[0], 0, c, s))
            for dete in self.deca[c].values() :
                heapq.heappush(red, (-self.maks[dete], 1, dete, None))
        return rezultat, poseceno

    def obrazac(self, obrazac : str, k : int = 10) :
        """Obicno stablo resava samo 'lov*', obrnuto samo '*ing'. Ostalo je greska."""
        oblik = oblik_obrasca(obrazac)
        moj = "obrnuto stablo" if self.obrnuto else "stablo"
        if oblik != moj :
            raise ValueError(f"'{obrazac}': {moj} ne resava ovaj oblik obrasca, koristite {oblik}.")
        return self.najcesca(obrazac.strip("*"), k)
