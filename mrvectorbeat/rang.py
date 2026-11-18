# Zajednicki korak za sva tri modela rangiranja (TF-IDF, kosinus, BM25).
#
# Svaki model vraca samo recnik pesma -> ocena (samo kandidati iz invertovanog
# indeksa). Ovde se ta ocene uredjuju u listu koju korisnik vidi. Istu funkciju
# zovu i strana Stihovi i evaluacija (scripts.pripremi), pa se meri tacno ona
# lista koja se prikazuje, bez posebne kopije rangiranja.


def uredi(ocene : dict, porodice = None) :
    """Sortira po oceni opadajuce (pri izjednacenju manji id pesme ide prvi) i
    od svake porodice zadrzava samo prvu, tj. najbolje ocenjenu verziju.
    Vraca (lista (pesma, ocena), broj ocenjenih pesama)."""
    redosled = sorted(ocene.items(), key = lambda x : (-x[1], x[0]))
    if porodice is None :
        return redosled, len(ocene)
    vidjene, rez = {}, []
    for d, o in redosled :
        f = porodice[d]
        if f in vidjene :
            continue
        vidjene[f] = True
        rez.append((d, o))
    return rez, len(ocene)


def rangiraj(ind, tfidf, bm25, upit : str, mera : str, porodice = None) :
    """Bira model po meri: "bm25" ide u BM25 model, "tfidf" i "kosinus" u TF-IDF
    model. Vraca isto sto i uredi()."""
    model = bm25 if mera == "bm25" else tfidf
    return uredi(model.ocene(ind, upit, mera), porodice)
