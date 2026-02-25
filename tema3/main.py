import numpy as np
import scipy.linalg


# ---------------------------------------------------------------------------
# Tema 3:
# Descompunerea QR a unei matrice patratice A folosind algoritmul Householder.
# Se rezolva sistemul Ax=b cu doua metode (Householder propriu + biblioteca),
# se calculeaza erorile si inversa matricei A prin descompunerea QR.
# ---------------------------------------------------------------------------


def solve_tema3(n=10, eps=1e-10):

    print(f"Running for n={n}, eps={eps}")

    # ==========================================
    # 1. GENERAREA DATELOR
    # ==========================================

    # Matrice A patratica n x n cu valori aleatoare in [0, 1)
    A_init = np.random.rand(n, n)

    # Vectorul s in R^n, ales aleator; acesta este solutia exacta a sistemului
    s = np.random.rand(n)

    # Calculam b_i = sum_{j=1}^{n} s_j * a_ij, adica b = A * s
    # Se itereaza explicit dupa formula din teorie (nu se foloseste @)
    b_init = np.zeros(n)
    for i in range(n):
        for j in range(n):
            b_init[i] += s[j] * A_init[i, j]

    # ==========================================
    # 2. DESCOMPUNEREA QR - ALGORITMUL HOUSEHOLDER
    # ==========================================
    # Ideea: se aplica (n-1) matrice de reflexie P_r astfel incat
    #   P_{n-1} * ... * P_1 * A = R  (matrice superior triunghiulara)
    # Matricea ortogonala Q^T = P_{n-1} * ... * P_1 se acumuleaza in Q_tilda.
    # La fiecare pas r se construieste P_r = I - (1/beta) * u * u^T.

    # Lucram pe copii pentru a nu modifica A_init si b_init
    A = A_init.copy()
    b = b_init.copy()

    # Initializam Q_tilda = I_n; la final Q_tilda va contine Q^T
    Q_tilda = np.eye(n)

    singular = False

    # Pasul r = 0, 1, ..., n-2  (indexare 0-based; corespunde r=1,...,n-1 din teorie)
    for r in range(n - 1):

        # sigma = sum_{j=r}^{n-1} a[j][r]^2  (suma patratelor pe coloana r, de la r in jos)
        sigma = 0.0
        for j in range(r, n):
            sigma += A[j, r] ** 2

        # Daca sigma <= eps, coloana r este deja nula sub diagonala => A singulara
        if sigma <= eps:
            singular = True
            print(f"ATENTIE: Matricea A este singulara la pasul r={r}!")
            break

        # k = sqrt(sigma); alegem semn(k) = -semn(a[r][r]) pentru stabilitate numerica
        # (evitam scaderea a doua numere aproape egale la calculul u[r] = a[r][r] - k)
        k = np.sqrt(sigma)
        if A[r, r] > 0:
            k = -k

        # beta = sigma - k * a[r][r]  =>  beta = ||u||^2 / 2  (cu u definit mai jos)
        beta = sigma - k * A[r, r]

        # Vectorul u folosit pentru constructia matricei de reflexie P_r:
        #   u[i] = 0,          pentru i = 0,...,r-1
        #   u[r] = a[r][r] - k
        #   u[i] = a[i][r],    pentru i = r+1,...,n-1
        u = np.zeros(n)
        u[r] = A[r, r] - k
        for i in range(r + 1, n):
            u[i] = A[i, r]

        # -- A = P_r * A --
        # Transformarea coloanelor j = r+1,...,n-1 (coloana r se seteaza direct mai jos)
        # Aplicarea P_r pe coloana j: a[i][j] -= gamma * u[i], i=r,...,n-1
        # unde gamma = (Ae_j, u) / beta = (sum_{i=r}^{n-1} u[i]*a[i][j]) / beta
        for j in range(r + 1, n):
            gamma = 0.0
            for i in range(r, n):
                gamma += u[i] * A[i, j]
            gamma /= beta
            for i in range(r, n):
                A[i, j] -= gamma * u[i]

        # Setam direct coloana r a lui R: a[r][r] = k, a[i][r] = 0 pentru i > r
        # (rezultatul exact al aplicarii P_r pe coloana r este acesta)
        A[r, r] = k
        for i in range(r + 1, n):
            A[i, r] = 0.0

        # -- b = P_r * b --
        # gamma = (b, u) / beta = (sum_{i=r}^{n-1} u[i]*b[i]) / beta
        # b[i] = b[i] - gamma * u[i], pentru i = r,...,n-1
        gamma = 0.0
        for i in range(r, n):
            gamma += u[i] * b[i]
        gamma /= beta
        for i in range(r, n):
            b[i] -= gamma * u[i]

        # -- Q_tilda = P_r * Q_tilda --
        # Acumulam transformarile pentru a obtine Q^T la final.
        # Se aplica aceeasi regula ca mai sus: q_tilda[i][j] -= gamma * u[i]
        # unde gamma = (Q_tilda_e_j, u) / beta = (sum_{i=r}^{n-1} u[i]*q_tilda[i][j]) / beta
        for j in range(n):
            gamma = 0.0
            for i in range(r, n):
                gamma += u[i] * Q_tilda[i, j]
            gamma /= beta
            for i in range(r, n):
                Q_tilda[i, j] -= gamma * u[i]

    # La finalul algoritmului:
    #   A       contine matricea R (superior triunghiulara)
    #   Q_tilda contine matricea Q^T
    #   b       contine Q^T * b_init (termenul liber transformat)
    R  = A        # R este matricea superior triunghiulara
    Qt = Q_tilda  # Qt = Q^T

    # Verificam ca R nu este singulara: |r_ii| > eps pentru toti i
    if not singular:
        for i in range(n):
            if abs(R[i, i]) <= eps:
                singular = True
                print(f"ATENTIE: Matricea A este singulara! R[{i},{i}] <= eps.")
                break

    # ==========================================
    # 3. REZOLVAREA SISTEMULUI Ax = b
    # ==========================================
    # Ax=b  <=>  QRx=b  <=>  Rx = Q^T*b
    # b deja contine Q^T * b_init dupa algoritmul Householder de mai sus.
    # Rezolvam sistemul superior triunghiular Rx = b prin substitutie inversa.

    # --- 3a. Solutia x_Householder (folosind descompunerea noastra) ---
    x_householder = np.zeros(n)
    if not singular:
        # Substitutie inversa: x[i] = (b[i] - sum_{j=i+1}^{n-1} R[i][j]*x[j]) / R[i][i]
        for i in range(n - 1, -1, -1):
            suma = 0.0
            for j in range(i + 1, n):
                suma += R[i, j] * x_householder[j]
            x_householder[i] = (b[i] - suma) / R[i, i]

    # --- 3b. Solutia x_QR (folosind descompunerea QR din biblioteca scipy) ---
    # scipy.linalg.qr returneaza Q si R astfel incat A_init = Q * R
    Q_lib, R_lib = scipy.linalg.qr(A_init)
    # Rx = Q^T * b_init; calculam mai intai Q^T * b_init
    Qt_lib_b = Q_lib.T @ b_init
    x_qr = np.zeros(n)
    # Substitutie inversa pe sistemul R_lib * x = Qt_lib_b
    for i in range(n - 1, -1, -1):
        suma = 0.0
        for j in range(i + 1, n):
            suma += R_lib[i, j] * x_qr[j]
        x_qr[i] = (Qt_lib_b[i] - suma) / R_lib[i, i]

    # Diferenta dintre cele doua solutii (ar trebui sa fie aproape de 0)
    diff_solutions = np.linalg.norm(x_qr - x_householder, ord=2)

    # ==========================================
    # 4. CALCULUL ERORILOR
    # ==========================================
    # Erorile masoara cat de bine solutia satisface sistemul original
    # si cat de aproape este de solutia exacta s.

    # ||A_init * x_householder - b_init||_2  (reziduul sistemului pt Householder)
    err_householder = np.linalg.norm(A_init @ x_householder - b_init, ord=2)

    # ||A_init * x_QR - b_init||_2  (reziduul sistemului pt biblioteca)
    err_qr = np.linalg.norm(A_init @ x_qr - b_init, ord=2)

    # ||x_householder - s||_2 / ||s||_2  (eroarea relativa fata de solutia exacta s)
    rel_err_householder = np.linalg.norm(x_householder - s, ord=2) / np.linalg.norm(s, ord=2)

    # ||x_QR - s||_2 / ||s||_2  (eroarea relativa fata de solutia exacta s)
    rel_err_qr = np.linalg.norm(x_qr - s, ord=2) / np.linalg.norm(s, ord=2)

    # ==========================================
    # 5. INVERSA MATRICEI A FOLOSIND QR HOUSEHOLDER
    # ==========================================
    # Coloana j a inversei A^{-1} se obtine rezolvand A*x = e_j
    # Adica: Rx = Q^T * e_j = coloana j din Q^T = Qt[:, j]
    # Se aplica substitutia inversa pentru fiecare coloana j = 1,...,n

    A_inv_householder = np.zeros((n, n))
    if not singular:
        for j in range(n):
            # b_col = coloana j a lui Q^T
            # Deoarece Qt = Q^T, Qt[:, j] reprezinta exact coloana j a lui Q^T
            b_col = Qt[:, j].copy()
            # Rezolvam R * x_col = b_col prin substitutie inversa
            x_col = np.zeros(n)
            for i in range(n - 1, -1, -1):
                suma = 0.0
                for jj in range(i + 1, n):
                    suma += R[i, jj] * x_col[jj]
                x_col[i] = (b_col[i] - suma) / R[i, i]
            # Memoram x_col ca si coloana j a inversei
            A_inv_householder[:, j] = x_col

    # Inversa calculata cu functia din biblioteca (pentru comparatie)
    A_inv_lib = np.linalg.inv(A_init)

    # ||A_inv_Householder - A_inv_bibl||_2  (diferenta dintre cele doua inverse)
    diff_inv = np.linalg.norm(A_inv_householder - A_inv_lib, ord=2)

    # ==========================================
    # 6. AFISAREA REZULTATELOR
    # ==========================================

    print(f"\n||x_QR - x_Householder||_2                       = {diff_solutions:.2e}")
    print(f"\n||A_init * x_Householder - b_init||_2            = {err_householder:.2e}")
    print(f"||A_init * x_QR - b_init||_2                     = {err_qr:.2e}")
    print(f"||x_Householder - s||_2 / ||s||_2                = {rel_err_householder:.2e}")
    print(f"||x_QR - s||_2 / ||s||_2                         = {rel_err_qr:.2e}")
    print(f"\n||A_inv_Householder - A_inv_bibl||_2             = {diff_inv:.2e}")

    # Verificam ca toate erorile relevante sunt sub pragul 10^-6
    threshold = 1e-6
    print(f"\nPrag acceptat: < {threshold}")
    ok = True
    for val, name in [
        (err_householder,     "||A_init*x_Householder - b_init||"),
        (err_qr,              "||A_init*x_QR - b_init||"),
        (rel_err_householder, "||x_Householder - s|| / ||s||"),
        (rel_err_qr,          "||x_QR - s|| / ||s||"),
    ]:
        if val >= threshold:
            print(f"  AVERTISMENT: {name} = {val:.2e} >= {threshold}")
            ok = False
    if ok:
        print("  -> REZULTAT CORECT: Toate erorile sunt mai mici decat 10^-6!")


# Apelam cu n aleator si eps standard
solve_tema3(n=15, eps=1e-10)
