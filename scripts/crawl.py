# Pokrece crawler i snima graf u data/graf_vikipedija.pkl.gz (ide uz projekat). Pokretanje iz korena:
#   python -u -m scripts.crawl              # svi izvodjaci, do MAX_STRANA stranica
#   python -u -m scripts.crawl --max 30     # kratka proba
# Preuzete stranice su u ~/mrvectorbeat_cache/wiki/, pa ponovno pokretanje ne salje iste
# zahteve (prekid je bezbedan: sledece pokretanje nastavlja iz kesa).
# Posle crawlera: python -m scripts.pripremi (PageRank i slicnost).

import sys
import time

from mrvectorbeat import pipeline, crawler

MAX_STRANA = 10000

max_strana = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else MAX_STRANA
df = pipeline.ucitaj_katalog()
# izvodjaci redom po broju pesama u katalogu (najzastupljeniji prvi), svi
imena = list(df["track_artist"].dropna().astype(str).str.strip().value_counts().index)
imena = [i for i in imena if i]

t0 = time.time()
c = crawler.Crawler()
print(f"User-Agent: {crawler.USER_AGENT}")
print(f"pauza izmedju zahteva: {c.pauza} s; izvodjaca iz kataloga: {len(imena)}; najvise stranica: {max_strana}")
graf = crawler.obidji(imena, max_strana, c)
pipeline.snimi_graf(graf)
s = graf["statistika"]
print(f"Gotovo za {time.time() - t0:.0f} s.")
print(f"  stranica u grafu (preuzetih clanaka): {s['stranica']}, grana: {s['grana']}")
print(f"  muzickih stranica: {s['muzickih']} ({100 * s['muzickih'] / max(1, s['stranica']):.1f}%)")
print(f"  izvodjaca iz kataloga sa clankom: {s['izvodjaca']} od {len(imena)}")
print(f"  poslatih zahteva: {s['zahteva']}, iz kesa: {s['iz_kesa']}")
