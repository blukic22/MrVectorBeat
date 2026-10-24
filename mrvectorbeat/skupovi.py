# Skupovni operatori nad postings listama.
#
# Ulaz je bilo koja rastuce sortirana sekvenca bez ponavljanja (list ili
# array('I')), izlaz je uvek Python lista, takodje rastuca. Svaka operacija
# je jedan prolaz sa dva pokazivaca: i ide kroz a, j kroz b, i uvek se
# pomera pokazivac na manjem elementu. Vreme je O(len(a) + len(b)).
# Ugradjeni skupovi i njihovi operatori se ovde namerno ne koriste.


def presek(a, b) -> list :
    """Elementi koji su i u a i u b."""
    i = j = 0
    rez = []
    while i < len(a) and j < len(b) :
        if a[i] == b[j] :
            rez.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j] :
            i += 1
        else :
            j += 1
    return rez


def unija(a, b) -> list :
    """Elementi koji su u a ili u b, svaki jednom."""
    i = j = 0
    rez = []
    while i < len(a) and j < len(b) :
        if a[i] == b[j] :
            rez.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j] :
            rez.append(a[i])
            i += 1
        else :
            rez.append(b[j])
            j += 1
    # jedna lista je potrosena, ostatak druge ide ceo
    rez.extend(a[i:])
    rez.extend(b[j:])
    return rez


def razlika(a, b) -> list :
    """Elementi iz a kojih nema u b."""
    i = j = 0
    rez = []
    while i < len(a) and j < len(b) :
        if a[i] == b[j] :
            i += 1
            j += 1
        elif a[i] < b[j] :
            rez.append(a[i])
            i += 1
        else :
            j += 1
    # ostatak a je veci od svega u b, pa sigurno nije u b
    rez.extend(a[i:])
    return rez


def presek_vise(liste) -> list :
    """Presek proizvoljnog broja lista. Ide od najkrace ka najduzoj, jer
    medjurezultat nikad nije duzi od najkrace liste, i staje cim je prazan."""
    if not liste :
        return []
    liste = sorted(liste, key = len)
    rez = list(liste[0])
    for lista in liste[1:] :
        if not rez :
            break
        rez = presek(rez, lista)
    return rez


def unija_vise(liste) -> list :
    """Unija proizvoljnog broja lista, spajanjem jedne po jedne."""
    rez = []
    for lista in liste :
        rez = unija(rez, lista)
    return rez
