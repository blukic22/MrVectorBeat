from mrvectorbeat.skupovi import presek

K = 3


def kgrami(kljuc : str, k : int = K) -> set :
    """Skup k-grama imena sa granicnikom $ na pocetku i kraju."""
    s = "$" + kljuc + "$"
    return {s[i:i + k] for i in range(len(s) - k + 1)}


def kgrami_obrasca(obrazac : str, k : int = K) -> set :
    """k-grami iz delova obrasca izmedju zvezdica. $ ide samo uz deo koji je
    stvarno na pocetku ili kraju imena (tj. obrazac tu nema zvezdicu)."""
    delovi = obrazac.split("*")
    grami = set()
    for i, d in enumerate(delovi) :
        if i == 0 and d :
            d = "$" + d
        if i == len(delovi) - 1 and d :
            d = d + "$"
        grami |= {d[j:j + k] for j in range(len(d) - k + 1)}
    return grami


def odgovara(kljuc : str, obrazac : str) -> bool :
    """Pravo poredjenje sa obrascem: prvi deo na pocetku, poslednji na kraju,
    srednji delovi redom, bez preklapanja."""
    delovi = obrazac.split("*")
    if not kljuc.startswith(delovi[0]) :
        return False
    poz = len(delovi[0])
    for d in delovi[1:-1] :
        i = kljuc.find(d, poz)
        if i < 0 :
            return False
        poz = i + len(d)
    poslednji = delovi[-1]
    return len(kljuc) - len(poslednji) >= poz and kljuc.endswith(poslednji)


class KgramIndeks :
    def __init__(self, imena) :
        """imena: lista (ime, tip, broj). id imena = njegov redni broj u listi."""
        self.imena = list(imena)
        self.kljucevi = [ime.lower() for ime, _, _ in self.imena]
        self.velicina = []      # velicina[id] = broj k-grama imena (za Zakarda)
        self.postings = {}      # k-gram -> rastuca lista id-jeva
        for id_, kljuc in enumerate(self.kljucevi) :
            grami = kgrami(kljuc)
            self.velicina.append(len(grami))
            for g in grami :
                self.postings.setdefault(g, []).append(id_)

    def obrazac(self, obrazac : str, k : int = 10) :
        """Imena koja odgovaraju obrascu sa zvezdicama, od najcesceg.
        Vraca (lista (broj, ime, tip), broj kandidata pre provere)."""
        obrazac = obrazac.lower()
        grami = kgrami_obrasca(obrazac)
        if not grami :
            raise ValueError(f"'{obrazac}': nijedan deo obrasca nema {K} znaka, "
                             f"k-gramski indeks nema po cemu da trazi.")
        liste = []
        for g in grami :
            if g not in self.postings :
                return [], 0, 0
            liste.append(self.postings[g])
        liste.sort(key = len)
        kandidati = liste[0]
        for lista in liste[1:] :
            kandidati = presek(kandidati, lista)
            if not kandidati :
                break
        # k-gramski indeks ne zna redosled ("be*t" pronalazi i "tbe..."), zato zavrsna provera
        tacni = [i for i in kandidati if odgovara(self.kljucevi[i], obrazac)]
        tacni.sort(key = lambda i : -self.imena[i][2])
        return [(self.imena[i][2], self.imena[i][0], self.imena[i][1]) for i in tacni[:k]], len(kandidati), len(tacni)

    def ispravka(self, unos : str, prag : float, n : int = 3) :
        """Najslicnija imena po Zakardovom koeficijentu k-grama.
        Kandidati su samo imena sa bar jednim zajednickim k-gramom (iz postings lista).
        Vraca (lista (zakard, ime, tip, broj) iznad praga, broj kandidata)."""
        a = kgrami(unos.lower())
        zajednicki = {}         # id -> |A ∩ B|
        for g in a :
            for i in self.postings.get(g, ()) :
                zajednicki[i] = zajednicki.get(i, 0) + 1
        ocene = []
        for i, z in zajednicki.items() :
            j = z / (len(a) + self.velicina[i] - z)      # |A ∪ B| = |A| + |B| - |A ∩ B|
            if j >= prag :
                ime, tip, broj = self.imena[i]
                ocene.append((j, ime, tip, broj))
        ocene.sort(key = lambda o : (-o[0], -o[3]))
        return ocene[:n], len(zajednicki)
