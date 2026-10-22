# app.py
#
# SABLON aplikacije. Bocna traka i sve strane vec postoje, ali su strane
# prazne - na svakoj je samo naslov i komentar koje gotove funkcije tu
# mozete koristiti.
#
# Funkcije za ucitavanje (ucitaj_X ispod) pozivaju pipeline.ucitaj_X().
# Dok su one u pipeline.py prazne, vracaju None - aplikacija se pokrece
# bez greske, samo jos nema sta da prikaze.
#
# VAZNO: ovo nije fiksno - imena funkcija, sta primaju i sta vracaju su
# samo PREDLOG, menjajte ih kako vam odgovara.
#
# Pokretanje iz korena projekta:
#   streamlit run app.py

import streamlit as st
import pandas as pd
from mrvectorbeat import pipeline, prikaz, tekst, autodopuna
from st_keyup import st_keyup
import math
import time

# OVDE IDU importi vasih modula, npr.:
#   from mrvectorbeat import tekst, upit, indeks, tfidf, bm25, ...

st.set_page_config(page_title="Mr Vectorbeat", layout="wide")
st.markdown(prikaz.CSS, unsafe_allow_html=True)


# --- Ucitavanje modela ---------------------------------------------------
#
# Svaka funkcija je obavijena dekoratorom da se ucita SAMO JEDNOM, a ne
# pri svakom kliku:
#   @st.cache_data      - za tabele i recnike (pandas DataFrame, dict)
#   @st.cache_resource  - za modele (objekti klasa, FAISS indeks, ...)

@st.cache_resource
def ucitaj_indeks():
    return pipeline.ucitaj_indeks()


@st.cache_resource
def ucitaj_trie():
    return pipeline.ucitaj_trie()


@st.cache_data
def ucitaj_katalog():
    return pipeline.ucitaj_katalog()


@st.cache_data
def ucitaj_izvodjace():
    pass


@st.cache_data
def broj_reci():
    pass


@st.cache_resource
def ucitaj_tfidf():
    return pipeline.ucitaj_tfidf()


@st.cache_resource
def ucitaj_bm25():
    return pipeline.ucitaj_bm25()


@st.cache_data
def ucitaj_evaluaciju():
    return pipeline.ucitaj_evaluaciju()


@st.cache_data
def ucitaj_qrels():
    return pipeline.ucitaj_qrels()


@st.cache_data
def ucitaj_graf():
    return pipeline.ucitaj_graf()


@st.cache_resource
def ucitaj_kgram():
    return pipeline.ucitaj_kgram()


@st.cache_data
def ucitaj_porodice():
    return pipeline.ucitaj_porodice()


@st.cache_data
def ucitaj_popularnost():
    return pipeline.ucitaj_popularnost()


@st.cache_resource
def ucitaj_kdrvo():
    return pipeline.ucitaj_kdrvo()


@st.cache_resource
def ucitaj_semantiku():
    return pipeline.ucitaj_semantiku()


@st.cache_resource
def ucitaj_rag_korpus():
    return pipeline.ucitaj_rag_korpus()


@st.cache_resource
def ucitaj_autoplay():
    return pipeline.ucitaj_autoplay()


@st.cache_data
def ucitaj_hibrid_evaluaciju():
    return pipeline.ucitaj_hibrid_evaluaciju()


@st.cache_resource
def ucitaj_sbert():
    pass


@st.cache_resource
def zagrej_ollama():
    pass


@st.cache_data
def ucitaj_labele_pesama():
    pass


@st.cache_data
def ucitaj_labele_autoplay():
    pass


@st.cache_data
def ucitaj_katalog_po_id():
    pass


@st.cache_resource
def ucitaj_jezicki(mu):
    pass


@st.cache_data
def ucitaj_tokene_po_pesmi():
    return pipeline.ucitaj_tokene()


@st.cache_data
def ucitaj_audio_vektore():
    pass


@st.cache_data
def ucitaj_sbert_vektore():
    pass

if not all(p.exists() for p in (pipeline.KATALOG, pipeline.TRIE, pipeline.KGRAM)):
    st.error("Nema artefakata. Pokrenite u terminalu: python -m scripts.pripremi")
    st.stop()

ind = ucitaj_indeks()
stablo, obrnuto_stablo = ucitaj_trie()
kgram_indeks = ucitaj_kgram()
porodice = ucitaj_porodice()
df = ucitaj_katalog()
izvodjaci = ucitaj_izvodjace()
# SBERT i Ollama se ucitavaju odmah pri pokretanju - trebaju i "Po smislu"
# i "AI DJ Bot", pa cekanje ide na start, a ne na prvi upit.
ucitaj_sbert()
zagrej_ollama()


# --- Bocna traka -------------------------------------------------------

with st.sidebar:
    st.markdown(prikaz.LOGO, unsafe_allow_html=True)

    strana = st.radio("Izaberite stranu", ["Obrada teksta", "Pretraga", "Stihovi", "Jezicki model", "Evaluacija",
                                           "Izvodjaci", "Katalog", "Radio", "Po smislu", "AI DJ Bot",
                                           "Hibridna pretraga", "Klasteri", "Autoplay"])

    st.divider()
    st.caption("Ucitani indeksi")
    # OVDE IDE statistika preko prikaz.statistika(labela, broj):
    #   prikaz.statistika("Pesama", len(df))
    #   prikaz.statistika("Izvodjaca", len(izvodjaci))
    #   prikaz.statistika("Reci u indeksu", broj_reci())


# --- Pretraga ----------------------------------------------------------

if strana == "Pretraga":
    st.title("Pretraga")
    st.caption("Kucajte naziv pesme ili izvodjaca.")

    st.caption("Zvezdica: lov* (stablo), *ing (obrnuto stablo), be*t, *ove*, s*ing* (k-gramski indeks).")

    # st_keyup salje vrednost posle svakog slova (debounce 200 ms), bez Enter-a
    unos = st_keyup("Ime pesme ili izvodjaca", key="pretraga_unos", debounce=200,
                    placeholder="npr. drake, lov*, *ing, be*t")

    t0 = time.perf_counter()
    r = autodopuna.predlozi(unos, stablo, obrnuto_stablo, kgram_indeks, k=10)
    ms = (time.perf_counter() - t0) * 1000

    if r["put"]:
        st.caption(f"Put: **{r['put']}** · {ms:.2f} ms · {r['info']}")
    if r["greska"]:
        st.error(r["greska"])
    elif r["predlozi"]:
        prefiks = unos.strip() if r["put"] == "stablo" else ""
        prefiks = prefiks.rstrip("*")
        for broj, ime, tip in r["predlozi"]:
            prikaz.sugestija(ime, prefiks, f"{tip} · {broj}×")
    elif r["put"] == "ispravka":
        if r["ispravka"]:
            j, ime, tip, broj = r["ispravka"][0]
            st.info(f"Nijedno ime ne pocinje sa „{unos.strip()}“. Da li ste mislili: "
                    f"**{ime}** ({tip}, Žakard {j:.2f})?")
            for j, ime, tip, broj in r["ispravka"][1:]:
                st.caption(f"ili: {ime} ({tip}, Žakard {j:.2f})")
        else:
            st.warning(f"Nema imena slicnog „{unos.strip()}“ (prag Žakarda {autodopuna.PRAG}).")
    elif r["put"]:
        st.warning("Nijedno ime ne odgovara obrascu.")


# --- Katalog -----------------------------------------------------------

elif strana == "Katalog":
    st.title("Katalog")

    c1, c2, c3, c4 = st.columns(4)
    zanr = c1.selectbox("Žanr", ["Svi"] + sorted(df["playlist_genre"].dropna().unique()))
    sort = c2.selectbox("Sortiraj po", ["Popularnost", "Godina izdanja", "Tempo"])
    opadajuce = c3.selectbox("Redosled", ["Opadajuće", "Rastuće"]) == "Opadajuće"
    po_strani = c4.selectbox("Pesama po strani", [12, 24, 48, 96], index=1)

    kolona = {"Popularnost": "track_popularity", "Godina izdanja": "godina", "Tempo": "tempo"}[sort]
    pogled = df if zanr == "Svi" else df[df["playlist_genre"] == zanr]
    pogled = pogled.sort_values(kolona, ascending=not opadajuce, na_position="last", kind="stable")

    ukupno = len(pogled)
    broj_strana = max(1, math.ceil(ukupno / po_strani))
    st.session_state["kat_strana"] = min(max(st.session_state.get("kat_strana", 1), 1), broj_strana)
    tekuca = st.number_input("Strana", 1, broj_strana, key="kat_strana")
    st.markdown(f"**Pesama koje odgovaraju izboru: {ukupno:,}** · strana **{tekuca}** od **{broj_strana}**")

    deo = pogled.iloc[(tekuca - 1) * po_strani: tekuca * po_strani]
    mreza = st.columns(4)
    for i, (_, red) in enumerate(deo.iterrows()):
        r = red.to_dict()
        r["godina"] = "—" if pd.isna(r["godina"]) else int(r["godina"])
        with mreza[i % 4]:
            prikaz.plocica(r)


# --- Stihovi -----------------------------------------------------------

elif strana == "Stihovi":
    st.title("Pretraga po stihovima")
    st.caption("Pretrazuje sadrzaj pesama, ne naslove.")

    # Koristi: ind, ucitaj_tfidf(), porodice, df, prikaz.kartica()


# --- Jezicki model -----------------------------------------------------

elif strana == "Jezicki model":
    st.title("Jezicki model i prosirenje upita")
    st.caption("Treci nacin rangiranja, i sta se sve moze dodati upitu.")

    # Koristi: ucitaj_jezicki(mu), ucitaj_bm25(), ucitaj_tokene_po_pesmi(),
    #          ucitaj_qrels(), ind, df, prikaz.kartica()


# --- Evaluacija --------------------------------------------------------

elif strana == "Evaluacija":
    st.title("Evaluacija pretrage")

    # Koristi: ucitaj_evaluaciju(), ucitaj_qrels(), ucitaj_tfidf(),
    #          ucitaj_bm25(), ind, df, prikaz.kartica()


# --- Izvodjaci ---------------------------------------------------------

elif strana == "Izvodjaci":
    st.title("Uticaj izvodjaca")
    st.caption("Ko je najuticajniji izvodjac - PageRank nad grafom koji je "
               "napravio crawler, naspram prostog brojanja linkova.")

    # Koristi: ucitaj_graf(), prikaz.kartica_izvodjaca()


# --- Radio -------------------------------------------------------------

elif strana == "Radio":
    st.title("Radio — slicne pesme po zvuku")
    st.caption("Pretraga po audio-osobinama (tempo, energija, glasnoca...), "
               "ne po tekstu ni zanru. k-d stablo + k-NN nad 9 osobina.")

    # Koristi: ucitaj_kdrvo(), ucitaj_labele_pesama(), ucitaj_katalog_po_id(),
    #          porodice, prikaz.kartica_radio()


# --- Po smislu ---------------------------------------------------------

elif strana == "Po smislu":
    st.title("Pretraga po smislu")
    st.caption("Opisi raspolozenje obicnim jezikom - sistem razume smisao, "
               "ne samo tacne reci iz stihova.")

    # Koristi: ucitaj_semantiku(), ucitaj_sbert(), ucitaj_katalog_po_id(),
    #          prikaz.kartica_smisao()


# --- AI DJ Bot ---------------------------------------------------------

elif strana == "AI DJ Bot":
    st.title("AI DJ Bot")
    st.caption("Pitaj o izvodjacima - bot odgovara na osnovu Wikipedia "
               "clanaka i pokazuje odakle je uzeo odgovor.")

    # Koristi: ucitaj_rag_korpus(), ucitaj_sbert()


# --- Klasteri ----------------------------------------------------------

elif strana == "Klasteri":
    st.title("Klasteri — grupisanje bez unapred poznatih oznaka")
    st.caption("K-means nad dva razlicita prostora: zvuk (9 audio-osobina) "
               "naspram smisla teksta (SBERT).")

    # Koristi: ucitaj_audio_vektore(), ucitaj_sbert_vektore(), df


# --- Autoplay ----------------------------------------------------------

elif strana == "Autoplay":
    st.title("Autoplay — predvidjanje sledece pesme")
    st.caption("Markovljev lanac naspram Transformera, oba obucena na "
               "stvarnoj istoriji slusanja (data/music4all_istorija.csv).")

    # Koristi: ucitaj_autoplay(), ucitaj_labele_autoplay(),
    #          ucitaj_katalog_po_id(), prikaz.kartica()


# --- Hibridna pretraga -------------------------------------------------

elif strana == "Hibridna pretraga":
    st.title("Hibridna pretraga — BM25 + SBERT (RRF)")
    st.caption("Isti upit kroz tacne reci (BM25) i kroz smisao (SBERT), pa "
               "spojeno preko Reciprocal Rank Fusion.")

    # Koristi: ucitaj_bm25(), ucitaj_semantiku(), ind, porodice,
    #          ucitaj_hibrid_evaluaciju(), prikaz.kartica()


# --- Obrada teksta -----------------------------------------------------

else:
    st.title("Obrada teksta")
    st.caption("Kako se stihovi pretvaraju u reci kojima pretrazujemo.")

    unos = st.text_area("Tekst", "I'm nothin' but a fool, and I dont care. Don't stop believin'!", height=130)
    r = tekst.obradi(unos)

    st.subheader("1. Mala slova")
    st.code(r["mala_slova"], language=None)

    st.subheader(f"2. Izdvajanje reci ({len(r['reci'])})")
    st.write(r["reci"])

    izbacene = [w for w in r["reci"] if w in tekst.stop_reci]
    st.subheader(f"3. Uklanjanje stop-reci ({len(r['bez_stop'])} ostalo)")
    st.write(r["bez_stop"])
    st.caption("Uklonjeno: " + (", ".join(izbacene) or "—"))

    st.subheader("4. Stemovanje")
    st.write(r["koreni"])
    st.dataframe(pd.DataFrame({"oblik": r["bez_stop"], "koren": r["koreni"]}).drop_duplicates(),
                 hide_index=True)
