# KONTEKST: radim projekat MrVectorbeat (muzički pretraživač), prva nedelja je ZAVRŠENA. Nastavi odatle.

Radi kao Claude Code: čitaj stvarne fajlove u projektu pre nego što pišeš kod, ne oslanjaj se na ovaj opis kao na izvor istine. Odgovaraj na srpskom (latinica). Objašnjavaj kratko i jednostavno, ja sam student i moram da odbranim kod.

## Okruženje
- Windows, PowerShell, koren projekta: C:\Users\Bogdan\OneDrive\Desktop\MrVectorbeat_template (kopija ima drugo ime)
- Venv se zove `venv` (ne .venv): `.\venv\Scripts\Activate.ps1`
- Pokretanje iz KORENA projekta: `python -m scripts.pripremi` (gradi artefakte, jednom) i `streamlit run app.py`
- Aplikacija samo UČITAVA artefakte iz artifacts/, nikad ne gradi pri kliku (@st.cache_data / @st.cache_resource za ucitaj_* funkcije)

## Pravila projekta (važe za sve nedelje)
- Dozvoljeno: streamlit, streamlit-keyup, pandas, pyarrow, numpy, nltk, standardna biblioteka. 
- nltk SAMO za listu stop-reči, stemer i WordNet. networkx SAMO za crtanje grafa u interfejsu.
- ZABRANJENO: scikit-learn, whoosh, rank_bm25, gensim, faiss, scipy i bilo koje gotove implementacije pretrage, indeksiranja, rangiranja i evaluacije. Sva logika mora biti ručno napisana.
- sentence-transformers, transformers, torch, faiss-cpu su u requirements.txt za kasnije delove, u prvom delu se ne koriste.
- Neće biti prihvaćeno: stemovanje svakog tokena iznova, posebni slučajevi u kodu za pojedine reči/pesme (npr. if rec == "don't"), strana Katalog koja čita sirov CSV.

## Skelet (zadat unapred)
- app.py: bočni meni sa stranama ["Obrada teksta","Pretraga","Stihovi","Jezicki model","Evaluacija","Izvodjaci","Katalog","Radio","Po smislu","AI DJ Bot","Hibridna pretraga","Klasteri","Autoplay"]; na vrhu su ucitaj_X() funkcije sa dekoratorima koje zovu pipeline.ucitaj_X() (neke su još `pass`); u komentarima svake strane piše šta se koristi (ind, tfidf, bm25, porodice, df, prikaz.kartica...). 
- mrvectorbeat/pipeline.py: parovi napravi_X()/ucitaj_X() za: katalog, tokene, trie, kgram, indeks, tfidf, bm25, evaluaciju (+ucitaj_qrels), jezički model, graf/rangove (PageRank), kdrvo, semantiku, rag_korpus, hibrid_evaluaciju, autoplay; plus ucitaj_porodice() (dict track_id->porodica) i ucitaj_popularnost() (dict ime->popularnost). Imena su samo predlog, smem da ih menjam.
- scripts/pripremi.py: poziva napravi_* redom (katalog, tokeni, trie, kgram, indeks, tfidf, bm25+evaluacija, jezicki, rangovi, kdrvo, semantika, rag, hibrid, autoplay); posle svakog poziva treba ispis šta je napravljeno. Redosled bitan.
- mrvectorbeat/prikaz.py: GOTOV, ne menjati. Sadrži CSS, LOGO, boja(), inicijali(), sugestija(ime, prefiks, tip), kartica(mesto, red, skor=None), statistika(labela, vrednost), plocica(red), kartica_radio(mesto, red, rastojanje, procenat), kartica_smisao(mesto, red, skor, nacin), kartica_izvodjaca(mesto, ime, skor, dolaznih). Kartice očekuju kolone: track_name, track_artist, godina, playlist_genre, tempo.
- Ulazni podaci u data/: spotify_songs.csv, qrels.csv, music4all_istorija.csv (i druge kasnije).

## NEDELJA 1 (zadatak, ispunjen)
1. Obrada teksta u 4 zasebna koraka (mala slova, izdvajanje reči, stop-reči, stemer), redosled se mora obrazložiti na odbrani.
2. Apostrof: reč sa apostrofom i bez njega daje isti termin; pravilo isto za stih, upit i listu stop-reči.
3. Obrada celog kataloga (>3M tokena), svaki različit oblik se stemuje tačno jednom (keš).
4. Zadržati samo engleske pesme sa stihovima.
5. Godina izdanja iz datuma u 3 oblika (2019-08-16, 2019-08, 2019).
6. Ključ porodice (remiks/remaster/live/feat. istog snimka = isti ključ), duplikati se ne brišu.
7. Artefakti: pročišćen katalog i obrađeni tokeni po pesmi.
8. Strana Katalog: mreža kartica, filter po žanru, sortiranje (popularnost/godina/tempo), broj pesama po strani, straničenje; prikaz ukupnog broja i trenutne strane; čita samo artefakt kataloga.

## Šta je urađeno (proveri u fajlovima)
**mrvectorbeat/tekst.py** (moj, pisao sam ga sam): apostrofi = "'\u2019"; skloni_apostrofe(tekst) (briše apostrof i spaja delove: don't->dont, nothin'->nothin); ucitaj_stop_reci() i `stop_reci` (frozenset, nltk lista provučena kroz isto skloni_apostrofe); mala_slova; izdvajanje_reci (briše apostrofe pa re.findall(r'[^\W_]+')); skloni_stop; klasa Stemer (SnowballStemmer("english"), rečnik `kes`, brojač `poziva_stemer`, metode koren i stemuj); globalni `stemer`; stemuj; obradi(tekst) vraća dict sa ključevima "mala_slova","reci","bez_stop","koreni"; termini(tekst) vraća listu korena (isti put za stih i upit).
Redosled i obrazloženje: mala slova -> reči -> stop-reči -> stemer. Stop-reči pre stemera jer nltk lista sadrži nestemovane oblike (stemer menja this->thi).
Svesna posledica pravila apostrofa: we'll->well, he'll->hell se poklapaju sa običnim rečima.

**mrvectorbeat/katalog.py**: izuci_godinu/izvuci_godine (split("-")[0], 4 cifre, vraća pd.NA, dtype Int64); skup OZNAKE (remaster, remastered, live, remix, mix, edit, edited, version, acoustic, ...), skup POCETNE {"feat","ft","featuring","with","from"}; _norm(s) (mala slova, bez apostrofa, vraća listu reči); je_oznaka(deo) (True ako deo počinje rečju iz POCETNE ili sadrži reč iz OZNAKE ili godinu 19xx/20xx); kljuc_porodice(naslov, izvodjac): naslov se seče regexom na " - " i zagrade, prvi deo se uvek zadržava, ostali samo ako nisu oznaka, + odsecanje "feat" bez zagrade; ključ je "izvodjac||naslov"; pomoćna pregled(df) za proveru ključa. Poznato ograničenje: deo naslova sa rečju iz OZNAKE (npr. "Song (Live Forever)") biva odbačen; izvođač se poredi tačno kako piše.

**mrvectorbeat/pipeline.py** (popunjeno): konstante CSV=data/spotify_songs.csv, ART=artifacts, KATALOG=artifacts/katalog.parquet, TOKENI=artifacts/tokeni.pkl.gz; KOLONE (track_id, track_name, track_artist, lyrics, track_album_name, godina, playlist_genre, playlist_subgenre, track_popularity, duration_ms, danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, porodica); napravi_katalog (filter language=="en", neprazni stihovi, drop_duplicates("track_id") jer je track_id jedinstven ključ za sve kasnije, godina, porodica, snima parquet); ucitaj_katalog; napravi_tokene (dict track_id -> lista termina, jedan KesiraniStemer za sve, assert poziva_stemer == len(kes), snima gzip pickle); ucitaj_tokene; ucitaj_porodice. Ostale funkcije u pipeline.py su još prazne (pass).

**scripts/pripremi.py**: napravi_katalog() i napravi_tokene() sa ispisima; ostali pozivi su još prazni šabloni.

**app.py**: popunjene strane "Obrada teksta" (prikazuje međurezultat svakog koraka) i "Katalog" (4 padajuće liste: žanr, sortiranje, redosled, broj po strani; number_input za stranu sa clamp-om preko st.session_state["kat_strana"]; brojač "Pesama koje odgovaraju izboru: N · strana X od Y"; mreža od 4 kolone preko prikaz.plocica; godina NA -> "—"). Dodata provera `if not pipeline.KATALOG.exists(): st.error(...); st.stop()` pre učitavanja. ucitaj_tokene_po_pesmi() vraća pipeline.ucitaj_tokene(). Ostale strane su prazne.

## Rezultati na pravim podacima (za odbranu)
- 18 454 redova posle filtera (en + neprazni stihovi) -> 15 405 pesama posle uklanjanja ponovljenih track_id; 12 688 porodica
- Tokeni: 3 360 835 tokena, 49 451 različitih oblika, 49 451 poziva stemera (jednaki = svaki oblik stemovan tačno jednom). Ranije je izlazilo 3 360 861 / 49 430 pre promene venva (verovatno različita verzija nltk liste stop-reči), nove brojke važe.
- Provera ključa porodice na pravim podacima: odbačeno su prave oznake (radio edit, remastered, remix...), najveće porodice su razumne (Lean On, Animals...).

## Šta treba
1. Prvo pročitaj stvarne fajlove (app.py, mrvectorbeat/*.py, scripts/pripremi.py, requirements.txt, data/) i potvrdi da se poklapaju sa opisom. Ako se razlikuju, važi kod u fajlovima.
2. Ne diraj ono što radi (Nedelja 1) osim ako nađeš grešku; ako nađeš, prvo mi je pokaži.
3. Čekaj da ti nalepim zadatak za Nedelju 2 (verovatno invertovani indeks u pipeline.py i strana Pretraga/Stihovi u app.py), pa ga razloži po fajlovima skeleta: šta ide u mrvectorbeat/ (novi moduli sa algoritmima), šta u pipeline.py (napravi_X/ucitaj_X), šta u pripremi.py (poziv + ispis), šta u app.py (strana).
4. Poštuj pravila: samo dozvoljene biblioteke, sve ručno napisano, bez specijalnih slučajeva u kodu, aplikacija samo učitava artefakte.
5. Posle svakog koraka daj komandu kojom proveravam da radi i ispis koji treba da vidim.