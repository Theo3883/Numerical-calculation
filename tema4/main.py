from pathlib import Path

def citeste_vector_din_fisier(nume_fisier):
    """Functie ajutatoare pentru a citi un vector dintr-un fisier text."""
    vector = []
    try:
        with open(nume_fisier, 'r') as f:
            for linie in f:
                # Fiecare linie contine un numar, separate cu endline
                valori = linie.strip().split()
                for val in valori:
                    vector.append(float(val))
    except FileNotFoundError:
        print(f"Eroare: Fisierul {nume_fisier} nu a fost gasit.")
    return vector

def rezolva_sistem_rar(fisier_d0, fisier_d1, fisier_d2, fisier_b, p_precizie):
    # Setăm precizia conform cerintei
    epsilon = 10 ** (-p_precizie)
    k_max = 10000
    
    # ---------------------------------------------------------
    # 1. Citire date si calcul dimensiune sistem (10 pct)
    # ---------------------------------------------------------
    d0 = citeste_vector_din_fisier(fisier_d0)
    d1 = citeste_vector_din_fisier(fisier_d1)
    d2 = citeste_vector_din_fisier(fisier_d2)
    b = citeste_vector_din_fisier(fisier_b)
    
    n = len(d0)
    if n == 0 or len(b) != n:
        print("Eroare la citirea datelor: dimensiunile d0 și b nu corespund.")
        return
        
    print(f"1. Dimensiunea sistemului (n): {n}")

    # ---------------------------------------------------------
    # 2. Găsire număr diagonale p și q (10 pct - continuare)
    # ---------------------------------------------------------
    p = n - len(d1)
    q = n - len(d2)
    print(f"2. Ordinele diagonalelor secundare sunt p = {p} și q = {q}")

    # ---------------------------------------------------------
    # 3. Verificare elemente d0 nenule (5 pct)
    # ---------------------------------------------------------
    for i in range(n):
        if abs(d0[i]) <= epsilon:
            print(f"3. Eroare: Elementul d0[{i}] este nul (sau mai mic decat epsilon). Sistemul nu se poate rezolva cu Gauss-Seidel.")
            return
    print("3. Verificare cu succes: toate elementele diagonalei principale sunt nenule.")

    # ---------------------------------------------------------
    # 4. Algoritmul Gauss-Seidel (35 pct)
    # ---------------------------------------------------------
    x_c = [0.0] * n # vectorul cu solutia la pasul curent
    x_p = [0.0] * n # vectorul cu solutia la pasul precedent
    
    k = 0
    print("4. Se ruleaza algoritmul Gauss-Seidel...")
    
    # faceum un do while
    while True:
        # Copiem x curent in x precedent
        for i in range(n):
            x_p[i] = x_c[i]
            
        # Calculam noul x_c
        for i in range(n):
            suma = 0.0
            
            # Diagonala inferioara p
            if i - p >= 0:
                suma += d1[i-p] * x_c[i-p]
                
            # Diagonala inferioara q
            if i - q >= 0:
                suma += d2[i-q] * x_c[i-q]
                
            # Diagonala superioara p
            if i + p < n:
                suma += d1[i] * x_p[i+p]
                
            # Diagonala superioara q
            if i + q < n:
                suma += d2[i] * x_p[i+q]
                
            # Actualizarea lui x_c[i] (formulele din document)
            x_c[i] = (b[i] - suma) / d0[i]
            
        # Calculam norma infinit a diferentei (delta x)
        delta_x = 0.0
        for i in range(n):
            dif = abs(x_c[i] - x_p[i])
            if dif > delta_x:
                delta_x = dif
                
        k += 1

        # Conditia de oprire a buclei while din document
        if not (delta_x >= epsilon and k <= k_max and delta_x <= 10**10):
            break

    # Verificare convergenta
    if delta_x < epsilon:
        print(f"   -> Solutia a convers dupa {k} iteratii.")
    else:
        print(f"   -> Divergenta! Algoritmul s-a oprit la iterația k = {k}.")
        print(f"      Ultima eroare (delta_x) a fost: {delta_x}")
        return
    # ---------------------------------------------------------
    # 5 & 6. Calcul ||A*x_GS - b|| (20 pct)
    # ---------------------------------------------------------
    norma_infinit = 0.0
    
    # Calculăm y = A * x_GS într-o singură parcurgere
    for i in range(n):
        y_i = d0[i] * x_c[i]
        
        if i - p >= 0:
            y_i += d1[i-p] * x_c[i-p]
        if i - q >= 0:
            y_i += d2[i-q] * x_c[i-q]
        if i + p < n:
            y_i += d1[i] * x_c[i+p]
        if i + q < n:
            y_i += d2[i] * x_c[i+q]
            
        # Calculăm diferența față de termenul liber b[i] și actualizăm norma
        diferenta = abs(y_i - b[i])
        if diferenta > norma_infinit:
            norma_infinit = diferenta

    print(f"5. Norma ||A*x_GS - b||_inf: {norma_infinit}")
    print(f"   (Norma este < epsilon? {'Da' if norma_infinit < epsilon else 'Nu'})")

# --------------------------------------------
# Rulam codul 
# --------------------------------------------
if __name__ == "__main__":
    # Parametrul 'p' din cerinta epsilon = 10^-p (de exemplu, 6 pentru epsilon = 1e-6)
    p_precizie = 6

    baza = Path(__file__).parent / "variables"
    index_set = 5

    fisier_d0 = baza / f"d0_{index_set}.txt"
    fisier_d1 = baza / f"d1_{index_set}.txt"
    fisier_d2 = baza / f"d2_{index_set}.txt"
    fisier_b = baza / f"b_{index_set}.txt"

    print(f"Rulez setul de date {index_set} din: {baza}")
    rezolva_sistem_rar(str(fisier_d0), str(fisier_d1), str(fisier_d2), str(fisier_b), p_precizie)