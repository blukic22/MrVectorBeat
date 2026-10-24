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

CSV = Path("data/spotify_songs.csv")
ART = Path("artifacts")
KATALOG = ART / "katalog.parquet"
TOKENI = ART / "tokeni.pkl.gz"
POZICIJE = ART / "pozicije.pkl.gz"
TRIE = ART / "trie.pkl.gz"
KGRAM = ART / "kgram.pkl.gz"
INDEKS = ART / "indeks.pkl.gz"

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
    """Gradi TF-IDF model i snima ga u artifacts/."""
    pass


def ucitaj_tfidf():
    """Cita gotov TF-IDF model iz artifacts/."""
    pass


def napravi_bm25():
    """Gradi BM25 model i snima ga u artifacts/."""
    pass


def ucitaj_bm25():
    """Cita gotov BM25 model iz artifacts/."""
    pass


def ucitaj_qrels(skup=None):
    """Cita test upite iz data/qrels.csv.

    skup - "poznata_stavka", "tema", ili None za oba.
    Vraca: tabelu (skup, upit, track_id, ocena, tipfeler)."""
    pass


def napravi_evaluaciju():
    """Racuna metrike za TF-IDF i BM25 nad qrels i snima tabelu u
    artifacts/."""
    pass


def ucitaj_evaluaciju():
    """Cita gotovu tabelu metrika iz artifacts/."""
    pass


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
