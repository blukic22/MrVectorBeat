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
from mrvectorbeat import pipeline, prikaz, tekst, autodopuna, rangirano, evaluacija
from st_keyup import st_keyup
import math
import time
import html

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


# cache_resource: isti objekat pri svakom renderu, bez kopije kataloga sa
# stihovima; strane ga samo citaju, nijedna ga ne menja
@st.cache_resource
def ucitaj_katalog():
    return pipeline.ucitaj_katalog()


@st.cache_data
def ucitaj_izvodjace():
    pass


@st.cache_data
def broj_reci():
    return len(ucitaj_indeks())


@st.cache_resource
def ucitaj_tfidf():
    return pipeline.ucitaj_tfidf()


@st.cache_resource
def ucitaj_bm25():
    return pipeline.ucitaj_bm25()


@st.cache_resource   # cele rang-liste (8 MB), cita se samo na strani Evaluacija
def ucitaj_evaluaciju():
    return pipeline.ucitaj_evaluaciju()


@st.cache_data
def ucitaj_qrels():
    return pipeline.ucitaj_qrels()


@st.cache_resource
def mere_evaluacije():
    """Mere za svaki upit i svaki model, nad rang-listama iz artefakta (ovde se
    nista ne rangira). Vraca (tabela mera po upitu, recnik upit -> ocene porodica)."""
    e = ucitaj_evaluaciju()
    ocene = pipeline.ocene_po_upitu(ucitaj_qrels())
    porodice = ucitaj_katalog()["porodica"].to_numpy()
    redovi = []
    for mera, po_upitu in e["liste"].items():
        for u in e["upiti"]:
            lista = [porodice[d] for d in po_upitu[u["upit"]]]
            m = evaluacija.mere_upita(lista, ocene.get(u["upit"], {}), evaluacija.prag(u["skup"]))
            redovi.append({"grupa": u["grupa"], "upit": u["upit"], "model": mera, **m})
    return pd.DataFrame(redovi), ocene


@st.cache_data
def ucitaj_graf():
    return pipeline.ucitaj_graf()


@st.cache_resource
def ucitaj_kgram():
    return pipeline.ucitaj_kgram()


@st.cache_resource   # isti recnik za svaki render, bez kopiranja posle svakog slova
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

if not all(p.exists() for p in (pipeline.KATALOG, pipeline.TRIE, pipeline.KGRAM, pipeline.INDEKS, pipeline.TFIDF, pipeline.BM25)):
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
    prikaz.statistika("Pesama", len(df))
    prikaz.statistika("Reci u indeksu", broj_reci())


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
    st.caption("Pozicije su mesta u listi svih reci stiha; stop-reci se ne indeksiraju, "
               "ali zauzimaju mesto.")

    PRAZAN = "Upit nema termina posle obrade: stop-reci se ne pretrazuju."
    SIRINA = 8          # reci levo i desno od pogotka u isecku

    def isecak(stih, pogoci):
        """Isecak oko prvog pogotka; pogoci je lista lista pozicija (jedan pogodak
        = pozicije koje se isticu). Pozicije su iste kao u indeksu jer su obe
        iz izdvajanje_reci nad istim stihom."""
        reci = tekst.izdvajanje_reci(stih)
        if not pogoci:
            od, do, istaknute = 0, 2 * SIRINA + 1, []
        else:
            od, do = max(0, min(pogoci[0]) - SIRINA), max(pogoci[0]) + SIRINA + 1
            istaknute = [p for pog in pogoci for p in pog]
        istaknute = set(istaknute)       # samo za prikaz, nije postings lista
        delovi = []
        for i in range(od, min(do, len(reci))):
            rec = html.escape(reci[i])   # prvo escape, pa tek onda <mark>
            delovi.append(f"<mark>{rec}</mark>" if i in istaknute else rec)
        tekst_isecka = ("… " if od > 0 else "") + " ".join(delovi) + (" …" if do < len(reci) else "")
        if len(pogoci) > 1:
            tekst_isecka += f" <span style='color:#8b8b98'>(još {len(pogoci) - 1} pogodaka)</span>"
        return tekst_isecka

    rezim = st.radio("Režim", ["Bulov", "Frazni", "Blizina", "Rangirano"], horizontal=True)
    poruka, pogodci, pozitivni = None, [], ""
    if rezim == "Rangirano" :
        rangirano.strana(ind, ucitaj_tfidf(), ucitaj_bm25(), df)
        st.stop()
    # pogodci: lista (id pesme, pogoci ili None); None znaci da se pozicije traze
    # tek za prikazane pesme (Bulov rezim)
    t0 = time.perf_counter()
    if rezim == "Bulov":
        op = st.radio("Operator", ["I", "ILI", "NE"], horizontal=True)
        if op == "NE":
            c1, c2 = st.columns(2)
            pozitivni = c1.text_input("Mora da sadrži", placeholder="npr. love")
            zabranjeni = c2.text_input("Ne sme da sadrži", placeholder="npr. hate cry")
            kljuc = (rezim, op, pozitivni, zabranjeni)
            uneto = pozitivni.strip() or zabranjeni.strip()
            if uneto and not tekst.termini(pozitivni) and not tekst.termini(zabranjeni):
                poruka = PRAZAN
            elif uneto:
                pogodci = [(d, None) for d in ind.bulov("NE", pozitivni, zabranjeni)]
        else:
            pozitivni = st.text_input("Termini", placeholder="npr. love heart")
            kljuc = (rezim, op, pozitivni)
            uneto = pozitivni.strip()
            if uneto and not tekst.termini(pozitivni):
                poruka = PRAZAN
            elif uneto:
                pogodci = [(d, None) for d in ind.bulov(op, pozitivni)]
    elif rezim == "Frazni":
        fraza = st.text_input("Fraza", placeholder="npr. the love of my life")
        kljuc = (rezim, fraza)
        uneto = fraza.strip()
        if uneto and not tekst.termini(fraza):
            poruka = PRAZAN
        elif uneto:
            pogodci = [(d, [list(range(p, p + duzina)) for p in pocetci])
                       for d, pocetci, duzina in ind.frazni(fraza)]
    else:
        c1, c2, c3 = st.columns([2, 2, 1])
        rec1 = c1.text_input("Prva reč", placeholder="npr. love")
        rec2 = c2.text_input("Druga reč", placeholder="npr. heart")
        k = c3.number_input("k (rastojanje < k)", 2, 50, 5)
        kljuc = (rezim, rec1, rec2, k)
        uneto = rec1.strip() and rec2.strip()
        n1, n2 = len(tekst.termini(rec1)), len(tekst.termini(rec2))
        if uneto and (n1 == 0 or n2 == 0):
            poruka = PRAZAN
        elif uneto and (n1 > 1 or n2 > 1):
            poruka = "Svako polje mora da se svede na tačno jedan termin."
        elif uneto:
            pogodci = [(d, [list(par) for par in parovi]) for d, parovi in ind.blizina(rec1, rec2, k)]
    ms = (time.perf_counter() - t0) * 1000

    if poruka:
        st.warning(poruka)
    elif uneto and not pogodci:
        st.info("Nema pesama koje odgovaraju upitu.")
    elif pogodci:
        # Redosled po popularnosti, pri izjednacenju po id-ju. Ovo NIJE rangiranje
        # (TF-IDF i BM25 su naredne nedelje): sortira se samo konacna lista
        # pogodaka, postings liste se nikad ne sortiraju.
        popularnost = df["track_popularity"].to_numpy()
        pogodci = sorted(pogodci, key=lambda x: (-popularnost[x[0]], x[0]))

        po_strani = 10
        broj_strana = max(1, math.ceil(len(pogodci) / po_strani))
        if st.session_state.get("stih_kljuc") != kljuc:    # novi upit kreće od prve strane
            st.session_state["stih_kljuc"] = kljuc
            st.session_state["stih_strana"] = 1
        st.session_state["stih_strana"] = min(max(st.session_state.get("stih_strana", 1), 1), broj_strana)
        tekuca = st.number_input("Strana", 1, broj_strana, key="stih_strana")
        st.markdown(f"**Pogodaka: {len(pogodci):,}** · strana **{tekuca}** od **{broj_strana}** · {ms:.1f} ms")

        deo = pogodci[(tekuca - 1) * po_strani: tekuca * po_strani]
        if rezim == "Bulov":
            # pozicije pozitivnih termina samo za prikazane pesme
            prikazane = sorted(d for d, _ in deo)
            po_terminu = [ind._pozicije_u(t, prikazane) for t in tekst.termini(pozitivni)]
            pozicije = {d: sorted(p for lista in po_terminu for p in lista[i])
                        for i, d in enumerate(prikazane)}
            deo = [(d, [[p] for p in pozicije[d]]) for d, _ in deo]

        for i, (d, pogoci) in enumerate(deo):
            red = df.iloc[d].to_dict()
            red["godina"] = "—" if pd.isna(red["godina"]) else int(red["godina"])
            prikaz.kartica((tekuca - 1) * po_strani + i + 1, red)
            st.markdown(f"<div style='margin:-4px 0 14px 8px'>{isecak(red['lyrics'], pogoci)}</div>",
                        unsafe_allow_html=True)


# --- Jezicki model -----------------------------------------------------

elif strana == "Jezicki model":
    st.title("Jezicki model i prosirenje upita")
    st.caption("Treci nacin rangiranja, i sta se sve moze dodati upitu.")

    # Koristi: ucitaj_jezicki(mu), ucitaj_bm25(), ucitaj_tokene_po_pesmi(),
    #          ucitaj_qrels(), ind, df, prikaz.kartica()


# --- Evaluacija --------------------------------------------------------

elif strana == "Evaluacija":
    st.title("Evaluacija pretrage")
    st.caption("Rang-liste za sve upite iz data/qrels.csv pravi scripts.pripremi, istom funkcijom "
               "kao strana Stihovi (jedna verzija po porodici, upiti sa greškom se ne ispravljaju). "
               "Ova strana samo čita te liste i računa mere.")
    if not pipeline.EVALUACIJA.exists():
        st.error("Nema artefakta evaluacije. Pokrenite: python -m scripts.pripremi")
        st.stop()

    ev = ucitaj_evaluaciju()
    tabela, ocene_upita = mere_evaluacije()
    NAZIVI = {v: k for k, v in rangirano.MERE.items()}       # "bm25" -> "BM25"

    # --- 1. Tabele: tri grupe, redovi su modeli, kolone cetiri mere ---
    st.subheader("Proseci po grupama upita")
    st.caption("Pogodak: za upit za jednu pesmu samo ocena 2 (sve verzije te pesme), za tematski "
               "ocena 1 ili 2. Ocena važi za celu porodicu. MAP deli brojem svih relevantnih "
               "pesama, nDCG@5 daje oceni 2 dvostruku težinu od ocene 1.")
    for g in evaluacija.GRUPE:
        deo = tabela[tabela["grupa"] == g]
        proseci = (deo.groupby("model", sort=False)[list(evaluacija.MERE)].mean()
                   .rename(index=NAZIVI))
        st.markdown(f"**{g}** · {deo['upit'].nunique()} upita · glavna mera: "
                    f"**{evaluacija.GLAVNA[g]}**")
        st.dataframe(proseci.style.format("{:.3f}").highlight_max(
            subset=[evaluacija.GLAVNA[g]], props="font-weight:bold"), width="stretch")

    # --- 2. Pregled jednog upita ---
    st.divider()
    st.subheader("Pregled upita")
    st.caption("Prvih 5 pesama svakog modela. :green[zeleno] ocena 2, :blue[plavo] ocena 1, "
               ":gray[sivo] ocena 0 (pesma nije u fajlu za ovaj upit).")
    izbor = st.selectbox("Upit iz fajla", list(range(len(ev["upiti"]))),
                         format_func=lambda i: f"{ev['upiti'][i]['grupa']} · {ev['upiti'][i]['upit']}")
    u = ev["upiti"][izbor]
    ocene = ocene_upita.get(u["upit"], {})
    porodice_kat = df["porodica"].to_numpy()
    BOJE = {2: "green", 1: "blue", 0: "gray"}

    def bez_markdowna(tekst_):
        # znakovi koje markdown tumaci (npr. ] zatvara :green[...], $ pokrece LaTeX)
        return "".join("\\" + z if z in "\\`*_[]{}$<>#~|" else z for z in str(tekst_))

    kolone = st.columns(len(ev["liste"]))
    for kol, (mera, po_upitu) in zip(kolone, ev["liste"].items()):
        lista = po_upitu[u["upit"]]
        kol.markdown(f"**{NAZIVI[mera]}** · {len(lista):,} pesama u listi")
        if not len(lista):
            kol.info("Nijedna pesma ne sadrži reči upita.")
        for i, d in enumerate(lista[:5], 1):
            o = ocene.get(porodice_kat[d], 0)
            kol.markdown(f":{BOJE[o]}[{i}. {bez_markdowna(df['track_name'].iat[d])} — "
                         f"{bez_markdowna(df['track_artist'].iat[d])}]")
        m = tabela[(tabela["upit"] == u["upit"]) & (tabela["model"] == mera)].iloc[0]
        kol.markdown(f"P@5 **{m['P@5']:.3f}** · RR **{m['MRR']:.3f}**  \n"
                     f"AP **{m['MAP']:.3f}** · nDCG@5 **{m['nDCG@5']:.3f}**")

    # --- 3. Parametri BM25 ---
    st.divider()
    st.subheader("Parametri BM25")
    st.caption("MRR nad upitima za jednu pesmu bez greške, za svaku kombinaciju k1 i b "
               "(računa scripts.pripremi). Jedna linija za svako b.")
    mreza = pd.DataFrame(ev["mreza"], columns=["k1", "b", "MRR"])
    mreza["b"] = "b = " + mreza["b"].map(lambda x: f"{x:.2f}")
    st.line_chart(mreza.pivot(index="k1", columns="b", values="MRR"), x_label="k1", y_label="MRR")
    najbolji = max(ev["mreza"], key=lambda x: x[2])
    izabran = next((r for r in ev["mreza"] if r[0] == ev["k1"] and r[1] == ev["b"]), None)
    c1, c2, c3 = st.columns(3)
    c1.metric("Izabrano k1", ev["k1"])
    c2.metric("Izabrano b", ev["b"])
    if izabran:
        c3.metric("MRR za izabrane", f"{izabran[2]:.3f}")
    st.markdown(f"Najbolja tačka mreže: k1 = {najbolji[0]}, b = {najbolji[1]} "
                f"(MRR {najbolji[2]:.3f}). Izabrane vrednosti koristi i strana Stihovi.")


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
