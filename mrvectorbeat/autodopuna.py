from mrvectorbeat.trie import oblik_obrasca

PRAG = 0.3      # najmanja Zakardova slicnost da bi se ispravka prikazala


def predlozi(unos : str, stablo, obrnuto, kgram, k : int = 10) :
    """Bira najbrzi put koji moze da resi unos. Sporiji putevi se pokrecu samo kad su potrebni:
    k-gramski indeks samo za zvezdicu koju stabla ne resavaju, ispravka samo kad dopuna ne vrati nista."""
    unos = unos.strip().lower()
    r = {"put" : None, "predlozi" : [], "ispravka" : [], "greska" : None, "info" : ""}
    if not unos :
        return r
    oblik = oblik_obrasca(unos)
    r["put"] = oblik
    try :
        if oblik == "stablo" :
            r["predlozi"], poseceno = stablo.obrazac(unos, k)
            r["info"] = f"poseceno cvorova: {poseceno} od {len(stablo)}"
        elif oblik == "obrnuto stablo" :
            r["predlozi"], poseceno = obrnuto.obrazac(unos, k)
            r["info"] = f"poseceno cvorova: {poseceno} od {len(obrnuto)}"
        else :
            r["predlozi"], kandidata, tacnih = kgram.obrazac(unos, k)
            r["info"] = f"kandidata iz preseka: {kandidata}, posle zavrsne provere: {tacnih}"
    except ValueError as e :
        r["greska"] = str(e)
        return r

    if not r["predlozi"] and "*" not in unos :
        r["put"] = "ispravka"
        r["ispravka"], kandidata = kgram.ispravka(unos, PRAG)
        r["info"] = f"kandidata sa bar jednim zajednickim k-gramom: {kandidata} od {len(kgram.imena)}"
    return r
