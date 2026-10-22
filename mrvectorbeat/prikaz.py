# mrvectorbeat/prikaz.py
#
# GOTOV fajl - izgled aplikacije (CSS, logo, kartice za rezultate).
# Ne morate ga menjati, samo pozivate njegove funkcije iz app.py.

import hashlib
import streamlit as st

BOJE = ["#8b3a3a", "#3a4a8b", "#3a6b52", "#6b4a7a", "#7a5a3a", "#3a6b7a"]

CSS = """
<style>
.kartica { display:flex; align-items:center; gap:14px; padding:12px 16px;
           border:1px solid #2a2a33; border-radius:10px; margin-bottom:8px; }
.mesto { color:#6b6b78; width:22px; text-align:center; }
.kvadrat { width:44px; height:44px; border-radius:8px; display:flex;
           align-items:center; justify-content:center; font-weight:700; color:#fff; }
.naslov { font-weight:600; font-size:16px; }
.podnaslov { color:#8b8b98; font-size:13px; margin-top:3px; }
.cip { background:#23232b; border-radius:5px; padding:2px 7px;
       font-size:12px; margin-left:6px; }
.sug { display:flex; justify-content:space-between; padding:8px 14px;
       border-bottom:1px solid #23232b; }
.tip { color:#6b6b78; font-size:12px; }
.reci { font-size: 19px; line-height: 2; letter-spacing: 0.3px;
        margin: 4px 0 30px 0; color: #d8d8dc; }
.rezultat { color: #32a852; font-weight: 600; }
</style>
"""

LOGO = """
<div style="display:flex; align-items:center; gap:12px; margin-bottom:2px;">
  <svg width="38" height="38" viewBox="0 0 38 38">
    <rect x="4"  y="14" width="4" height="10" rx="2" fill="#e2413f">
      <animate attributeName="height" values="10;26;10" dur="1.1s"
               repeatCount="indefinite" begin="0s"/>
      <animate attributeName="y" values="14;6;14" dur="1.1s"
               repeatCount="indefinite" begin="0s"/>
    </rect>
    <rect x="11" y="10" width="4" height="18" rx="2" fill="#e2413f">
      <animate attributeName="height" values="18;8;18" dur="1.1s"
               repeatCount="indefinite" begin="0.2s"/>
      <animate attributeName="y" values="10;15;10" dur="1.1s"
               repeatCount="indefinite" begin="0.2s"/>
    </rect>
    <rect x="18" y="6"  width="4" height="26" rx="2" fill="#e2413f">
      <animate attributeName="height" values="26;12;26" dur="1.1s"
               repeatCount="indefinite" begin="0.4s"/>
      <animate attributeName="y" values="6;13;6" dur="1.1s"
               repeatCount="indefinite" begin="0.4s"/>
    </rect>
    <rect x="25" y="12" width="4" height="14" rx="2" fill="#e2413f">
      <animate attributeName="height" values="14;24;14" dur="1.1s"
               repeatCount="indefinite" begin="0.6s"/>
      <animate attributeName="y" values="12;7;12" dur="1.1s"
               repeatCount="indefinite" begin="0.6s"/>
    </rect>
  </svg>
  <div>
    <div style="font-size:26px; font-weight:800; line-height:1.05;">
      MrVectorbeat
      <span style="font-size:11px; color:#e2413f; font-weight:600;
                   vertical-align:super;">v1.0</span>
    </div>
    <div style="color:#8b8b98; font-size:13px;">Muzicki pretrazivac</div>
  </div>
</div>
"""


def boja(ime):
    """Uvek ista boja za isto ime."""
    broj = int(hashlib.md5(ime.encode()).hexdigest(), 16)
    return BOJE[broj % len(BOJE)]


def inicijali(ime):
    delovi = ime.split()
    if len(delovi) >= 2:
        return (delovi[0][0] + delovi[1][0]).upper()
    return ime[:2].upper()


def sugestija(ime, prefiks, tip):
    """Jedan red u padajucoj listi, sa podebljanim delom koji se poklapa."""
    if ime.lower().startswith(prefiks.lower()):
        n = len(prefiks)
        tekst_ime = f"<b style='color:#e2413f'>{ime[:n]}</b>{ime[n:]}"
    else:
        tekst_ime = ime

    st.markdown(
        f"<div class='sug'><span>{tekst_ime}</span>"
        f"<span class='tip'>{tip}</span></div>",
        unsafe_allow_html=True,
    )


def kartica(mesto, red, skor=None):
    """Jedan rezultat pretrage."""
    ime = red["track_name"]
    if skor is None:
        cip_skor = ""
    else:
        cip_skor = f"<span class='cip'>skor {skor:.3f}</span>"
    st.markdown(
        f"<div class='kartica'>"
        f"<div class='mesto'>{mesto}</div>"
        f"<div class='kvadrat' style='background:{boja(ime)}'>{inicijali(ime)}</div>"
        f"<div><div class='naslov'>{ime}</div>"
        f"<div class='podnaslov'>{red['track_artist']} · {red['godina']}"
        f"<span class='cip'>{red['playlist_genre']}</span>"
        f"<span class='cip'>{round(red['tempo'])} BPM</span>"
        f"{cip_skor}"
        f"</div></div></div>",
        unsafe_allow_html=True,
    )


def statistika(labela, vrednost):
    levo, desno = st.columns([2, 1])
    levo.markdown(f"<span style='color:#8b8b98'>{labela}</span>", unsafe_allow_html=True)
    desno.markdown(
        f"<div style='text-align:right;font-weight:600'>{vrednost:,}</div>".replace(",", " "),
        unsafe_allow_html=True,
    )


def plocica(red):
    """Kvadratna kartica za mrezu kataloga."""
    ime = red["track_name"]
    st.markdown(
        f"<div style='border:1px solid #23232b; border-radius:12px;"
        f"padding:14px; margin-bottom:12px; height:190px;'>"
        f"<div style='height:74px; border-radius:8px; background:{boja(ime)};"
        f"display:flex; align-items:center; justify-content:center;"
        f"font-size:24px; font-weight:800; color:#fff; margin-bottom:10px;'>"
        f"{inicijali(ime)}</div>"
        f"<div style='font-weight:600; font-size:14px; line-height:1.3;"
        f"overflow:hidden; text-overflow:ellipsis; white-space:nowrap;'>{ime}</div>"
        f"<div style='color:#8b8b98; font-size:12px; margin-top:4px;"
        f"overflow:hidden; text-overflow:ellipsis; white-space:nowrap;'>"
        f"{red['track_artist']}</div>"
        f"<div style='color:#6b6b78; font-size:11px; margin-top:6px;'>"
        f"{red['godina']} · {red['playlist_genre']}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def kartica_radio(mesto, red, rastojanje, procenat):
    """Jedan rezultat na strani Radio, sa rastojanjem i procentom
    slicnosti."""
    ime = red["track_name"]
    st.markdown(
        f"<div class='kartica'>"
        f"<div class='mesto'>{mesto}</div>"
        f"<div class='kvadrat' style='background:{boja(ime)}'>{inicijali(ime)}</div>"
        f"<div><div class='naslov'>{ime}</div>"
        f"<div class='podnaslov'>{red['track_artist']} · {red['godina']}"
        f"<span class='cip'>{red['playlist_genre']}</span>"
        f"<span class='cip'>{procenat:.0f}% slicno</span>"
        f"<span class='cip'>rastojanje {rastojanje:.3f}</span>"
        f"</div></div></div>",
        unsafe_allow_html=True,
    )


def kartica_smisao(mesto, red, skor, nacin):
    """Jedan rezultat pretrage po smislu; skor od 0 do 1."""
    ime = red["track_name"]
    st.markdown(
        f"<div class='kartica'>"
        f"<div class='mesto'>{mesto}</div>"
        f"<div class='kvadrat' style='background:{boja(ime)}'>{inicijali(ime)}</div>"
        f"<div><div class='naslov'>{ime}</div>"
        f"<div class='podnaslov'>{red['track_artist']} · {red['godina']}"
        f"<span class='cip'>{red['playlist_genre']}</span>"
        f"<span class='cip'>{max(0.0, skor) * 100:.0f}% slicno</span>"
        f"<span class='cip'>{nacin}</span>"
        f"</div></div></div>",
        unsafe_allow_html=True,
    )


def kartica_izvodjaca(mesto, ime, skor, dolaznih):
    """Jedan red rang-liste izvodjaca."""
    st.markdown(
        f"<div class='kartica'>"
        f"<div class='mesto'>{mesto}</div>"
        f"<div class='kvadrat' style='background:{boja(ime)}'>"
        f"{inicijali(ime)}</div>"
        f"<div><div class='naslov'>{ime}</div>"
        f"<div class='podnaslov'>"
        f"<span class='cip'>rang {skor:.4f}</span>"
        f"<span class='cip'>{dolaznih} dolaznih</span>"
        f"</div></div></div>",
        unsafe_allow_html=True,
    )
