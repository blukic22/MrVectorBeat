# mrvectorbeat/pipeline.py
#
# SABLON. Ovde su samo IMENA funkcija i opis sta svaka treba da radi -
# tela su prazna (pass), pa se svaka funkcija moze pozvati bez greske,
# ali jos nista ne radi.
#
# VAZNO: ovo NISU fiksne funkcije. Imena, parametri i ono sto funkcija
# vraca su samo PREDLOG - slobodno ih menjajte, dodajte nove ili
# izbacite one koje vam ne trebaju, kako vam odgovara.
#
# Ideja ovog fajla: za svaki model postoji par funkcija
#   napravi_X() - gradi model JEDNOM i cuva ga u artifacts/
#                 (poziva je scripts/pripremi.py)
#   ucitaj_X()  - cita vec sacuvan model iz artifacts/
#                 (poziva je app.py)
# Tako aplikacija nikad ne gradi nista iznova pri svakom kliku.

# OVDE IDU importi vasih modula, npr.:
#   from mrvectorbeat import tekst, trie, indeks, ...

import gzip, pickle
from array import array
from collections import Counter
import pandas as pd
from pathlib import Path
from mrvectorbeat import tekst, katalog
from mrvectorbeat.trie import Trie
from mrvectorbeat.kgram import KgramIndeks, K
from mrvectorbeat.indeks import Indeks
from mrvectorbeat.tfidf import TfidfModel
from mrvectorbeat.bm25 import Bm25Model
from mrvectorbeat import rang, evaluacija
import time

CSV = Path("data/spotify_songs.csv")
ART = Path("artifacts")
KATALOG = ART / "katalog.parquet"
TOKENI = ART / "tokeni.pkl.gz"
POZICIJE = ART / "pozicije.pkl.gz"
TRIE = ART / "trie.pkl.gz"
KGRAM = ART / "kgram.pkl.gz"
INDEKS = ART / "indeks.pkl.gz"
TFIDF = ART / "tfidf.pkl.gz"
BM25 = ART / "bm25.pkl.gz"
EVALUACIJA = ART / "evaluacija.pkl.gz"
QRELS = Path("data/qrels.csv")

# Evaluacija: tri modela i mreza parametara BM25 za grafik MRR(k1) po b
MERE_EVALUACIJE = ("tfidf", "kosinus", "bm25")
MREZA_K1 = [i / 4 for i in range(0, 13)]        # 0, 0.25, ..., 3.0
MREZA_B = [0.0, 0.25, 0.5, 0.75, 1.0]

# Kolone koje zadrzavamo u katalogu. Sve ostalo iz CSV-a nam ne treba.
# Kartice u prikaz.py ocekuju bar: track_name, track_artist, godina,
# playlist_genre, tempo.
KOLONE = [
    "track_id",           # jedinstveni ID pesme, kljuc za sve kasnije
    "track_name",
    "track_artist",
    "lyrics",
    "track_album_name",
    "godina",             # ovu kolonu pravimo mi
    "playlist_genre",
    "playlist_subgenre",
    "track_popularity",
    "duration_ms",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "porodica",           # ovu kolonu pravimo mi - isti kljuc za razlicite
                          # verzije iste pesme
]


def napravi_katalog():
    podaci = pd.read_csv(CSV)
    n0 = len(podaci)
    podaci = podaci[podaci["language"] == "en"]
    podaci = podaci[podaci["lyrics"].fillna("").str.strip() != ""]
    podaci = podaci.drop_duplicates("track_id").copy()
    podaci["godina"] = katalog.izvuci_godine(podaci["track_album_release_date"])
    podaci["porodica"] = [katalog.kljuc_porodice(t, a) for t, a in zip(podaci["track_name"], podaci["track_artist"])]
    df = podaci[KOLONE].reset_index(drop = True)
    ART.mkdir(exist_ok = True)
    df.to_parquet(KATALOG, index = False)
    print(f"podaci {n0} -> {len(df)} pesama, {df['porodica'].nunique()} porodica")
    return df

def ucitaj_katalog():
    return pd.read_parquet(KATALOG)


def napravi_tokene():
    """Snima dva artefakta iz jednog prolaza: tokeni (track_id -> lista termina,
    format iz Nedelje 1) i pozicije (track_id -> array('I'), pozicija svakog
    termina u listi svih reci stiha, paralelno sa tokenima)."""
    df = ucitaj_katalog()
    stemer = tekst.Stemer()
    tokeni, pozicije, ukupno = {}, {}, 0
    for tid, stih in zip(df["track_id"], df["lyrics"]):
        koreni, poz = tekst.termini_sa_pozicijama(stih, stemer)
        assert len(poz) == len(koreni)
        tokeni[tid] = koreni
        pozicije[tid] = array("I", poz)
        ukupno += len(koreni)
    assert stemer.poziva_stemer == len(stemer.kes)
    with gzip.open(TOKENI, "wb") as f:
        pickle.dump(tokeni, f, protocol=pickle.HIGHEST_PROTOCOL)
    with gzip.open(POZICIJE, "wb") as f:
        pickle.dump(pozicije, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  tokena: {ukupno}, razlicitih oblika: {len(stemer.kes)}, "
          f"poziva stemera: {stemer.poziva_stemer}")
    print(f"  pozicija: {sum(len(p) for p in pozicije.values())}")
    return tokeni


def ucitaj_tokene():
    with gzip.open(TOKENI, "rb") as f:
        return pickle.load(f)


def ucitaj_pozicije():
    """Vraca recnik track_id -> array('I') pozicija, paralelno sa tokenima."""
    with gzip.open(POZICIJE, "rb") as f:
        return pickle.load(f)


def imena_iz_kataloga(df):
    """Lista (ime, tip, broj): svako ime pesme i izvodjaca jednom, sa brojem
    pojavljivanja u katalogu. Velika i mala slova se ne razlikuju; prikazuje
    se najcesci originalni oblik."""
    imena = []
    for tip, kolona in (("pesma", "track_name"), ("izvođač", "track_artist")):
        broj, oblici = Counter(), {}
        for ime in df[kolona].dropna():
            ime = str(ime).strip()
            if not ime:
                continue
            kljuc = ime.lower()
            broj[kljuc] += 1
            oblici.setdefault(kljuc, Counter())[ime] += 1
        for kljuc, n in broj.items():
            imena.append((oblici[kljuc].most_common(1)[0][0], tip, n))
    return imena


def napravi_trie():
    df = ucitaj_katalog()
    stablo, obrnuto = Trie(), Trie(obrnuto = True)
    for ime, tip, broj in imena_iz_kataloga(df):
        stablo.dodaj(ime, tip, broj)
        obrnuto.dodaj(ime, tip, broj)
    with gzip.open(TRIE, "wb") as f:
        pickle.dump((stablo, obrnuto), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  stablo: {len(stablo)} cvorova, obrnuto stablo: {len(obrnuto)} cvorova")
    return stablo, obrnuto


def ucitaj_trie():
    """Vraca (stablo, obrnuto stablo)."""
    with gzip.open(TRIE, "rb") as f:
        return pickle.load(f)


def ucitaj_porodice():
    """Vraca recnik track_id -> porodica."""
    df = pd.read_parquet(KATALOG, columns=["track_id", "porodica"])
    return dict(zip(df["track_id"], df["porodica"]))


def ucitaj_popularnost():
    """Vraca recnik ime (pesma ili izvodjac) -> popularnost."""
    pass


def napravi_kgram():
    df = ucitaj_katalog()
    ind = KgramIndeks(imena_iz_kataloga(df))
    with gzip.open(KGRAM, "wb") as f:
        pickle.dump(ind, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  imena: {len(ind.imena)}, razlicitih {K}-grama: {len(ind.postings)}")
    return ind


def ucitaj_kgram():
    with gzip.open(KGRAM, "rb") as f:
        return pickle.load(f)


def napravi_indeks():
    """Gradi pozicioni invertovani indeks iz tokena i pozicija i snima ga u
    artifacts/. Id pesme = redni broj reda u katalogu."""
    df = ucitaj_katalog()
    tokeni, pozicije = ucitaj_tokene(), ucitaj_pozicije()
    ids = list(df["track_id"])
    ind = Indeks([tokeni[t] for t in ids], [pozicije[t] for t in ids], ids)
    with gzip.open(INDEKS, "wb") as f:
        pickle.dump(ind, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  termina: {len(ind)}, postings unosa: {ind.broj_postings()}, "
          f"pozicija: {ind.broj_pozicija()}, fajl: {INDEKS.stat().st_size / 2**20:.1f} MB")
    return ind


def ucitaj_indeks():
    with gzip.open(INDEKS, "rb") as f:
        return pickle.load(f)


def napravi_tfidf():
    """Gradi TF-IDF model iz pozicionog indeksa (tf = odmak[i+1] - odmak[i]) i snima ga u artifacts/."""
    t0 = time.perf_counter()
    ind = ucitaj_indeks()
    model = TfidfModel(ind)
    with gzip.open(TFIDF, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  termina: {len(model.idf)}, tezina uz postings: {model.broj_tezina()}, "
          f"fajl: {TFIDF.stat().st_size / 2**20:.1f} MB, gradnja: {time.perf_counter() - t0:.1f} s")
    return model


def ucitaj_tfidf():
    with gzip.open(TFIDF, "rb") as f:
        return pickle.load(f)


def napravi_bm25():
    """Gradi BM25 model iz pozicionog indeksa: idf, duzina svake pesme i
    prosecna duzina se racunaju jednom i snimaju u artifacts/."""
    t0 = time.perf_counter()
    ind = ucitaj_indeks()
    model = Bm25Model(ind)
    with gzip.open(BM25, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  termina: {len(model.idf)}, k1 = {model.k1}, b = {model.b}, "
          f"prosecna duzina pesme: {model.avgdl:.1f} termina, "
          f"fajl: {BM25.stat().st_size / 2**20:.1f} MB, gradnja: {time.perf_counter() - t0:.1f} s")
    return model


def ucitaj_bm25():
    with gzip.open(BM25, "rb") as f:
        return pickle.load(f)


def ucitaj_qrels(skup=None):
    """Cita test upite iz data/qrels.csv i dodaje kolonu porodica (iz kataloga).

    skup - "poznata_stavka", "tema", ili None za oba.
    Vraca: tabelu (skup, upit, track_id, ocena, tipfeler, porodica)."""
    q = pd.read_csv(QRELS)
    kat = pd.read_parquet(KATALOG, columns=["track_id", "porodica"])
    q = q.merge(kat, on="track_id", how="left")
    if skup is not None:
        q = q[q["skup"] == skup].reset_index(drop=True)
    return q


def ocene_po_upitu(q):
    """Recnik upit -> {porodica: ocena}. Ocena iz fajla vazi za celu porodicu;
    ako bi porodica imala vise ocena, uzima se najveca."""
    rez = {}
    for upit, f, o in zip(q["upit"], q["porodica"], q["ocena"]):
        if pd.isna(f):
            continue
        ocene = rez.setdefault(upit, {})
        ocene[f] = max(ocene.get(f, 0), int(o))
    return rez


def napravi_evaluaciju():
    """Pusta sve upite iz qrels kroz sva tri modela, istom funkcijom kao strana
    Stihovi (rang.rangiraj, jedna verzija po porodici), i cuva CELE rang-liste.
    Upiti sa greskom se ne ispravljaju. Racuna i MRR za mrezu k1 x b nad upitima
    za jednu pesmu bez greske (grafik na strani Evaluacija)."""
    t0 = time.perf_counter()
    df = ucitaj_katalog()
    porodice = df["porodica"].to_numpy()
    ind, tfidf, bm25 = ucitaj_indeks(), ucitaj_tfidf(), ucitaj_bm25()
    q = ucitaj_qrels()
    upiti = q.drop_duplicates("upit")[["skup", "upit", "tipfeler"]].to_dict("records")
    for u in upiti:
        u["grupa"] = evaluacija.grupa(u["skup"], u["tipfeler"])

    liste = {}
    for mera in MERE_EVALUACIJE:
        liste[mera] = {}
        for u in upiti:
            rez, _ = rang.rangiraj(ind, tfidf, bm25, u["upit"], mera, porodice)
            liste[mera][u["upit"]] = array("I", (d for d, _ in rez))
    t1 = time.perf_counter()

    ocene = ocene_po_upitu(q)
    bez_greske = [u["upit"] for u in upiti if u["grupa"] == evaluacija.GRUPE[0]]
    mreza = []
    for b in MREZA_B:
        for k1 in MREZA_K1:
            zbir = 0.0
            for upit in bez_greske:
                rez, _ = rang.uredi(bm25.ocene(ind, upit, "bm25", k1, b), porodice)
                zbir += evaluacija.reciprocni_rang([porodice[d] for d, _ in rez], ocene.get(upit, {}), 2)
            mreza.append((k1, b, zbir / len(bez_greske)))
    najbolji = max(mreza, key=lambda x: x[2])

    rezultat = {"upiti": upiti, "liste": liste, "mreza": mreza, "k1": bm25.k1, "b": bm25.b}
    with gzip.open(EVALUACIJA, "wb") as f:
        pickle.dump(rezultat, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  upita: {len(upiti)} ({', '.join(f'{g}: {sum(u['grupa'] == g for u in upiti)}' for g in evaluacija.GRUPE)}), "
          f"modela: {len(MERE_EVALUACIJE)}, rang-liste: {t1 - t0:.1f} s")
    print(f"  mreza k1 x b: {len(MREZA_K1)} x {len(MREZA_B)} = {len(mreza)} tacaka, {time.perf_counter() - t1:.1f} s")
    print(f"  najbolji MRR na mrezi: {najbolji[2]:.4f} za k1 = {najbolji[0]}, b = {najbolji[1]}; "
          f"izabrano u bm25.py: k1 = {bm25.k1}, b = {bm25.b}")
    print(f"  fajl: {EVALUACIJA.stat().st_size / 2**20:.1f} MB")
    return rezultat


def ucitaj_evaluaciju():
    """Recnik: upiti (skup, upit, tipfeler, grupa), liste (mera -> upit ->
    array id-jeva pesama), mreza (k1, b, MRR), izabrani k1 i b."""
    with gzip.open(EVALUACIJA, "rb") as f:
        return pickle.load(f)


def napravi_jezicki(mu=1000.0):
    """Gradi jezicki model i snima ga u artifacts/."""
    pass


def ucitaj_jezicki():
    """Cita gotov jezicki model iz artifacts/."""
    pass


def ucitaj_graf():
    """Cita graf izvodjaca koji je napravio vas crawler."""
    pass


def napravi_rangove():
    """Racuna PageRank nad grafom izvodjaca i snima rang-listu u
    artifacts/."""
    pass


def ucitaj_rangove():
    """Cita gotovu rang-listu iz artifacts/."""
    pass


def napravi_kdrvo():
    """Gradi k-d stablo nad audio-osobinama kataloga i snima ga u
    artifacts/."""
    pass


def ucitaj_kdrvo():
    """Cita gotovo k-d stablo iz artifacts/."""
    pass


def napravi_semantiku():
    """Gradi semanticki model nad katalogom i snima ga u artifacts/."""
    pass


def ucitaj_semantiku():
    """Cita gotov semanticki model iz artifacts/."""
    pass


def napravi_rag_korpus():
    """Gradi korpus za AI DJ Bot-a i snima ga u artifacts/."""
    pass


def ucitaj_rag_korpus():
    """Cita gotov RAG korpus iz artifacts/."""
    pass


def napravi_hibrid_evaluaciju():
    """Racuna metrike hibridne pretrage nad qrels i snima tabelu u
    artifacts/."""
    pass


def ucitaj_hibrid_evaluaciju():
    """Cita gotovu tabelu metrika hibridne pretrage iz artifacts/."""
    pass


def napravi_autoplay():
    """Gradi Autoplay nad data/music4all_istorija.csv i snima ga u
    artifacts/."""
    pass


def ucitaj_autoplay():
    """Cita gotov Autoplay iz artifacts/."""
    pass
