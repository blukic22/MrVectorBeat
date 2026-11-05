import html
import math
import time

import pandas as pd
import streamlit as st

from mrvectorbeat import prikaz, tekst

MERE = {"TF-IDF": "tfidf", "Kosinus (lnc.ltc)": "kosinus"}
PO_STRANI = 10
SIRINA = 8                  
PRAZAN = "Upit nema termina posle obrade: stop-reci se ne pretrazuju."


def _red(df, d) :
    red = df.iloc[d].to_dict()
    red["godina"] = "—" if pd.isna(red["godina"]) else int(red["godina"])
    return red


def _isecak(stih, pozicije) :
    reci = tekst.izdvajanje_reci(stih)
    if pozicije:
        od, do = max(0, pozicije[0] - SIRINA), pozicije[-1] + SIRINA + 1
        do = min(do, pozicije[0] + 3 * SIRINA)
    else:
        od, do = 0, 2 * SIRINA + 1
    istaknute = {p for p in pozicije}            # samo za prikaz, nije postings lista
    delovi = []
    for i in range(od, min(do, len(reci))):
        rec = html.escape(reci[i])               # prvo escape, pa tek onda <mark>
        delovi.append(f"<mark>{rec}</mark>" if i in istaknute else rec)
    return ("… " if od > 0 else "") + " ".join(delovi) + (" …" if do < len(reci) else "")


def _istakni_sve(stih, koreni) :
    """Ceo stih; istaknute su reci cijim korenom se bavi tabela doprinosa."""
    koreni_stiha, poz = tekst.termini_sa_pozicijama(stih)
    istaknute = {p for k, p in zip(koreni_stiha, poz) if k in koreni}
    reci = tekst.izdvajanje_reci(stih)
    return " ".join(f"<mark>{html.escape(r)}</mark>" if i in istaknute else html.escape(r)
                    for i, r in enumerate(reci))


def strana(ind, model, df) :
    st.caption("Pesme poređane po oceni sličnosti sa upitom. Ocenjuju se samo pesme koje "
               "sadrže bar jednu reč iz upita. TF-IDF je zbir težina (1 + log tf) · idf; "
               "kosinus (lnc.ltc) normalizuje dužinu pesme i upita.")
    porodice = df["porodica"].to_numpy()

    c1, c2 = st.columns([3, 2])
    upit = c1.text_input("Upit", placeholder="npr. love heart", key="rang_upit")
    mera_naziv = c2.radio("Mera sličnosti", list(MERE), horizontal=True, key="rang_mera")
    mera = MERE[mera_naziv]

    if not upit.strip():
        return
    termini = tekst.termini(upit)
    if not termini:
        st.warning(PRAZAN)
        return

    t0 = time.perf_counter()
    rezultat, ocenjeno = model.rangiraj(ind, upit, mera, porodice)
    ms = (time.perf_counter() - t0) * 1000
    if not rezultat:
        st.info("Nema pesama koje odgovaraju upitu.")
        return

    broj_strana = max(1, math.ceil(len(rezultat) / PO_STRANI))
    kljuc = (upit, mera)
    if st.session_state.get("rang_kljuc") != kljuc:      
        st.session_state["rang_kljuc"] = kljuc
        st.session_state["rang_strana"] = 1
    st.session_state["rang_strana"] = min(max(st.session_state.get("rang_strana", 1), 1), broj_strana)
    tekuca = st.number_input("Strana", 1, broj_strana, key="rang_strana")
    st.markdown(f"**Pesama u rezultatu: {len(rezultat):,}** · ocenjeno **{ocenjeno:,}** od "
                f"{len(df):,} (samo one sa rečima iz upita) · strana **{tekuca}** od **{broj_strana}** "
                f"· {ms:.1f} ms")

    deo = rezultat[(tekuca - 1) * PO_STRANI: tekuca * PO_STRANI]
    prikazane = sorted(d for d, _ in deo)
    razliciti = list(dict.fromkeys(termini))
    po_terminu = [ind._pozicije_u(t, prikazane) for t in razliciti]
    pozicije = {d: sorted(p for lista in po_terminu for p in lista[i]) for i, d in enumerate(prikazane)}

    for i, (d, ocena) in enumerate(deo):
        red = _red(df, d)
        prikaz.kartica((tekuca - 1) * PO_STRANI + i + 1, red, ocena)
        st.markdown(f"<div style='margin:-4px 0 6px 8px'>{_isecak(red['lyrics'], pozicije[d])}</div>",
                    unsafe_allow_html=True)
        if st.button("Slične pesme", key=f"sl_{d}"):
            st.session_state["rang_slicna"] = d
            st.session_state.pop("rang_uporedi", None)

    d0 = st.session_state.get("rang_slicna")
    if d0 is None:
        return
    st.divider()
    r0 = _red(df, d0)
    st.subheader(f"Slične pesme: {r0['track_name']} — {r0['track_artist']}")
    st.caption("Kosinus (ltc.ltc) između tekstova. Poredi se samo sa pesmama koje dele reči; "
               "verzije iste pesme su izostavljene.")
    slicne = model.slicne(ind, d0, 10, porodice)
    if not slicne:
        st.info("Nema pesama sa sličnim tekstom.")
        return
    for i, (p, kosinus) in enumerate(slicne):
        prikaz.kartica(i + 1, _red(df, p), kosinus)
        if st.button("Uporedi", key=f"up_{d0}_{p}"):
            st.session_state["rang_uporedi"] = (d0, p)

    par = st.session_state.get("rang_uporedi")
    if not par or par[0] != d0:
        return
    a, b = par
    ukupno, tabela = model.doprinosi(a, b, 10)
    st.divider()
    st.subheader("Poređenje tekstova")
    st.markdown(f"**Kosinusna sličnost: {ukupno:.3f}** · istaknute su reči iz tabele (deset reči koje "
                f"najviše doprinose sličnosti)")
    koreni = {r[0] for r in tabela}
    levo, desno = st.columns(2)
    for kolona, d in ((levo, a), (desno, b)):
        r = _red(df, d)
        kolona.markdown(f"**{html.escape(r['track_name'])}** — {html.escape(r['track_artist'])}",
                        unsafe_allow_html=True)
        kolona.markdown("<div style='max-height:420px; overflow-y:auto; line-height:1.8'>"
                        f"{_istakni_sve(r['lyrics'], koreni)}</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Reč (koren)": [r[0] for r in tabela],
        "Pojavljivanja u levoj pesmi": [r[1] for r in tabela],
        "Pojavljivanja u desnoj pesmi": [r[2] for r in tabela],
        "Udeo u sličnosti (%)": [round(100 * r[4], 1) for r in tabela],
    }), hide_index=True)