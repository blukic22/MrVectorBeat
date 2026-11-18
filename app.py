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
from mrvectorbeat import pipeline, prikaz, tekst, autodopuna, rangirano, evaluacija, jezicki, plejlista
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


@st.cache_resource   # PageRank i slicnost; None dok crawler i pripremi ne naprave graf
def ucitaj_izvodjace():
    return pipeline.ucitaj_rangove() if pipeline.RANGOVI.exists() else None


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


@st.cache_resource
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
def ucitaj_jezicki():
    return pipeline.ucitaj_jezicki()


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
    if not pipeline.JEZICKI.exists():
        st.error("Nema jezičkog modela. Pokrenite: python -m scripts.pripremi")
        st.stop()
    jm = ucitaj_jezicki()
    tfidf_model = ucitaj_tfidf()
    porodice_kat = df["porodica"].to_numpy()
    PRIKAZANO = 10
    oznake = st.session_state.setdefault("jm_oznake", {})     # pesma -> 1 (dopada) / -1 (ne dopada)

    # --- Koja je ovo pesma? ---
    st.subheader("Koja je ovo pesma?")
    st.caption(f"Jezički model sa Dirichlet ublažavanjem (μ = {jm.mu:g}): pesme su poređane po "
               "log P(upit | pesma). Pesma kojoj fale neke reči ostaje u rezultatu, samo niže. "
               "Ako najboljoj pesmi fale reči, one se proširuju WordNet sinonimima manje težine.")
    upit = st.text_input("Deo stiha kojeg se sećate", placeholder="npr. found love in a place without hope",
                         key="jm_upit")
    rezultat, prikazane = [], []
    if upit.strip():
        t0 = time.perf_counter()
        r = jezicki.pretraga(ind, jm, upit, porodice_kat)
        ms = (time.perf_counter() - t0) * 1000
        rezultat = r["rezultat"]
        if not r["reci"]:
            st.warning("Upit nema reči posle obrade: stop-reči se ne pretražuju.")
        elif not rezultat:
            st.info("Nijedna pesma ne sadrži nijednu reč upita.")
        else:
            if r["prosireno"]:
                st.info("Uključeni su i rezultati za: " + ", ".join(s for s, _, _ in r["dodate"]))
            st.markdown(f"**Pesama u rezultatu: {len(rezultat):,}** · ocenjeno **{r['ocenjeno']:,}** od "
                        f"{len(df):,} (samo one sa bar jednom reči) · {ms:.1f} ms")
            deo = rezultat[:PRIKAZANO]
            prikazane = [d for d, _ in deo]
            otkucane = list(dict.fromkeys(k for _, k in r["reci"]))
            oblik = {}
            for rec, k in r["reci"]:
                oblik.setdefault(k, rec)
            for s, k, _ in r["dodate"]:
                oblik[k] = s
            pokriva = jezicki.pokrivenost(ind, otkucane + [k for _, k, _ in r["dodate"]], sorted(prikazane))
            for i, (d, ocena) in enumerate(deo, 1):
                prikaz.kartica(i, df.iloc[d].to_dict() | {"godina": "—" if pd.isna(df["godina"].iat[d])
                                                              else int(df["godina"].iat[d])}, ocena)
                ima = [oblik[k] for k in otkucane if pokriva[d][k]]
                fali = [oblik[k] for k in otkucane if not pokriva[d][k]]
                sin = [s for s, k, _ in r["dodate"] if pokriva[d][k]]
                opis = "Sadrži: " + (", ".join(ima) or "—")
                if fali:
                    opis += " · **Nedostaje: " + ", ".join(fali) + "**"
                if sin:
                    opis += " · sinonimi: " + ", ".join(sin)
                c0, c1, c2 = st.columns([4, 1.3, 1.5])
                c0.markdown(html.escape(opis))
                o = oznake.get(d, 0)
                if c1.button(("✓ " if o == 1 else "") + "Dopada mi se", key=f"jm_da_{d}"):
                    if o == 1:
                        oznake.pop(d)                     # drugi klik uklanja oznaku
                    else:
                        oznake[d] = 1
                    st.rerun()
                if c2.button(("✓ " if o == -1 else "") + "Ne dopada mi se", key=f"jm_ne_{d}"):
                    if o == -1:
                        oznake.pop(d)
                    else:
                        oznake[d] = -1
                    st.rerun()

    # --- Plejlista po mom ukusu ---
    st.divider()
    st.subheader("Plejlista po mom ukusu")
    dopada = sorted(d for d, o in oznake.items() if o == 1)
    ne_dopada = sorted(d for d, o in oznake.items() if o == -1)
    st.caption(f"Označeno: {len(dopada)} dopada se, {len(ne_dopada)} ne dopada se. Rocchio: "
               f"{plejlista.BETA} · prosek pesama koje se dopadaju − {plejlista.GAMA} · prosek onih koje "
               "se ne dopadaju (ltc vektori teksta).")
    c1, c2 = st.columns([1, 1])
    if c1.button("Napravi plejlistu", type="primary"):
        st.session_state["jm_plejlista"] = True
    if oznake and c2.button("Ukloni sve oznake"):
        oznake.clear()
        st.rerun()
    if st.session_state.get("jm_plejlista"):
        # racuna se iz TRENUTNIH oznaka pri svakom prikazu, pa promena oznake menja plejlistu
        if not dopada and not ne_dopada:
            dopada = sorted(prikazane[:plejlista.PODRAZUMEVANO])
            if dopada:
                st.caption(f"Nema oznaka: pretpostavlja se da vam se dopadaju prva "
                           f"{len(dopada)} rezultata pretrage.")
        if not dopada:
            st.info("Označite bar jednu pesmu koja vam se dopada (ili pretražite nešto).")
        else:
            p = plejlista.napravi(ind, tfidf_model, dopada, ne_dopada, prikazane, porodice_kat)
            if not p["plejlista"]:
                st.info("Nema dovoljno sličnih pesama za plejlistu.")
            else:
                oblici = plejlista.oblici(df, dopada, [t for t, _ in p["reci"]])
                st.markdown("**Napravljena na osnovu reči:** " + ", ".join(
                    html.escape(oblici[t]) for t, _ in p["reci"]))
                for i, (d, ocena) in enumerate(p["plejlista"], 1):
                    prikaz.kartica(i, df.iloc[d].to_dict() | {"godina": "—" if pd.isna(df["godina"].iat[d])
                                                                  else int(df["godina"].iat[d])}, ocena)

    # --- izbor mu ---
    with st.expander("Izbor parametra μ"):
        st.caption("MRR nad upitima za jednu pesmu iz qrels (bez proširenja), za više vrednosti μ; "
                   "računa scripts.pripremi.")
        mreza_mu = pd.DataFrame(jm.mreza, columns=["μ", "grupa", "MRR"]).pivot(
            index="μ", columns="grupa", values="MRR")
        st.line_chart(mreza_mu, x_label="μ", y_label="MRR")
        st.dataframe(pd.DataFrame({g: {"bez proširenja": a, "sa proširenjem": b}
                                   for g, (a, b) in jm.prosirenje.items()}).T.style.format("{:.3f}"))
        st.markdown(f"Izabrano μ = **{jm.mu:g}**. Za vrlo malo μ MRR je malo veći, ali tada skoro "
                    "samo broj pogođenih reči odlučuje i refren koji se ponavlja ne pomaže: "
                    "„found love in a place without hope” daje Rihannu tek na 7. mestu za μ = 5, "
                    "a na 1. od μ = 100.")


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
    if izvodjaci is None:
        st.error("Nema grafa i rangova. Pokrenite: python -u -m scripts.crawl, pa python -m scripts.pripremi")
        st.stop()
    graf = ucitaj_graf()
    naslovi = graf["naslovi"]
    ime = lambda a: graf["izvodjac"][a][0]           # ime iz kataloga
    s = izvodjaci["statistika"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stranica u grafu", f"{s['stranica']:,}")
    c2.metric("Muzičkih", f"{100 * s['muzickih'] / max(1, s['stranica']):.1f}%")
    c3.metric("Linkova (grana)", f"{s['grana']:,}")
    c4.metric("Izvođača iz kataloga", f"{len(izvodjaci['poredak']):,}")

    tab1, tab2 = st.tabs(["Najuticajniji izvođači", "Slični izvođači"])

    # --- najuticajniji ---
    with tab1:
        st.caption(f"PageRank nad celim grafom (d = {izvodjaci['d']}, {izvodjaci['iteracija']} iteracija); "
                   "prikazani su samo izvođači iz kataloga. Uticaj zavisi i od toga koliko su važne "
                   "stranice koje pokazuju na izvođača, a ne samo od njihovog broja.")
        poredak = izvodjaci["poredak"]
        broj = st.select_slider("Koliko izvođača", [10, 20, 50, 100], value=20, key="izv_broj")
        trazi = st.text_input("Nađi izvođača", key="izv_trazi", placeholder="npr. Queen").strip().lower()
        mesto = {a: i for i, a in enumerate(poredak, 1)}
        prikaz_izv = [a for a in poredak if trazi in ime(a).lower()] if trazi else poredak
        for a in prikaz_izv[:broj]:
            prikaz.kartica_izvodjaca(mesto[a], ime(a), izvodjaci["pr"][a], izvodjaci["dolaznih"][a])
        if trazi and not prikaz_izv:
            st.info("Nema tog izvođača u grafu (možda za njega nije nađen članak).")
        with st.expander("Izbor faktora prigušenja d i poređenje sa brojanjem linkova"):
            top = poredak[:20]
            redovi = []
            for d, o in izvodjaci["osetljivost"].items():
                redovi.append({"d": d, "iteracija": o["iteracija"],
                               "zajedničkih u top 20 sa d = 0.85": len(set(o["poredak"][:20]) & set(top))})
            redovi.append({"d": izvodjaci["d"], "iteracija": izvodjaci["iteracija"],
                           "zajedničkih u top 20 sa d = 0.85": 20})
            st.dataframe(pd.DataFrame(redovi).sort_values("d"), hide_index=True)
            st.markdown(f"Samo broj dolaznih linkova: zajedničkih u top 20 sa PageRank-om "
                        f"**{len(set(izvodjaci['po_dolaznim'][:20]) & set(top))}**.")
            st.markdown("d = 0.85: surfer u 85% slučajeva prati link, u 15% skače na slučajnu stranicu. "
                        "Manje d (0.5) gura poredak ka ravnomernom i skoro briše razliku među stranicama, "
                        "veće d (0.95) daje veći uticaj zatvorenim grupama stranica i sporije konvergira "
                        "(više iteracija).")

    # --- slicni ---
    with tab2:
        sl = izvodjaci["slicnost"]
        st.caption("Dva izvođača su slična ako na njih pokazuju iste stranice. Stranica koja pokazuje na "
                   "mnogo izvođača (spisak nagrada) vredi manje: težina log(A / broj izvođača na koje "
                   "pokazuje); sličnost je kosinus.")
        po_imenu = sorted(sl.izvodjaci, key=lambda a: ime(a).lower())
        a = st.selectbox("Izvođač koga volite", po_imenu, format_func=ime, key="izv_izabran",
                         index=po_imenu.index(poredak[0]) if poredak else 0)
        slicni = sl.slicni(a, 10)
        if not slicni:
            st.info("Na ovog izvođača ne pokazuje nijedna stranica koja pokazuje i na druge izvođače.")
        for i, (b, kos, stranice, ukupno) in enumerate(slicni, 1):
            prikaz.kartica_izvodjaca(i, ime(b), kos, izvodjaci["dolaznih"][b])
            ostalo = f" (i još {ukupno - len(stranice)})" if ukupno > len(stranice) else ""
            st.markdown(f"<div style='margin:-4px 0 8px 8px'>Oboje se pominju u: "
                        f"{html.escape(', '.join(naslovi[p] for p in stranice))}{ostalo}</div>",
                        unsafe_allow_html=True)
        if slicni:
            st.caption("Na kartici je kosinusna sličnost (umesto ranga).")
            import matplotlib.pyplot as plt
            import networkx as nx
            G = nx.Graph()                                  # networkx samo za crtanje
            for b, _, stranice, _ in slicni[:6]:
                for p in stranice[:3]:
                    G.add_edge(ime(a), naslovi[p])
                    G.add_edge(ime(b), naslovi[p])
            izv_imena = {ime(a)} | {ime(b) for b, _, _, _ in slicni[:6]}
            fig, ax = plt.subplots(figsize=(9, 6))
            poz = nx.spring_layout(G, seed=7, k=0.9)
            nx.draw_networkx_nodes(G, poz, ax=ax, node_size=[900 if n == ime(a) else 500 if n in izv_imena
                                                             else 200 for n in G],
                                   node_color=["#e4572e" if n == ime(a) else "#4e79a7" if n in izv_imena
                                               else "#bab0ac" for n in G])
            nx.draw_networkx_edges(G, poz, ax=ax, alpha=0.4)
            nx.draw_networkx_labels(G, poz, ax=ax, font_size=8)
            ax.axis("off")
            st.pyplot(fig)
            st.caption("Crveno: izabrani izvođač, plavo: slični, sivo: stranice koje ih povezuju.")


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
