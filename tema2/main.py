import numpy as np
import scipy.linalg

def solve_tema2(n=101, eps=1e-8):

    # ==========================================
    # 1. GENERAREA DATELOR
    # ==========================================

    print(f"Running for n={n} eps={eps}")
    
    B = np.random.rand(n, n)
    A = np.dot(B, B.T) 
    
    # Generam vectorul termenilor liberi b
    b = np.random.rand(n)

    # Salvam diagonala originala a lui A inainte sa o stricam
    diag_A_original = np.diag(A).copy()

    # ==========================================
    # 2. REZOLVARE CU BIBLIOTECA MATEMATICA
    # ==========================================
    # Folosim scipy.linalg.lu_factor si lu_solve pentru descompunerea LU
    lu, piv = scipy.linalg.lu_factor(A)
    x_lib = scipy.linalg.lu_solve((lu, piv), b)

    # ==========================================
    # 3. DESCOMPUNEREA LDL^T (In-place)
    # ==========================================
    
    for p in range(n):
        # Calculam elementul diagonal d_p
        sum_dp = sum(A[k, k] * A[p, k]**2 for k in range(p))
        A[p, p] = A[p, p] - sum_dp
        
        # Inlocuim if(v!=0) cu verificarea abs(v) > eps
        if abs(A[p, p]) <= eps:
            raise ValueError(f"Impartire la zero prevenita! A[{p}, {p}] este prea mic (<= eps).")
            
        # Calculam elementele de pe coloana p a matricei L
        for i in range(p + 1, n):
            sum_lip = sum(A[k, k] * A[i, k] * A[p, k] for k in range(p))
            A[i, p] = (A[i, p] - sum_lip) / A[p, p]
            # Elementul proaspat calculat se salveaza direct in A (sub diagonala principala)

    # ==========================================
    # 4. CALCUL DETERMINANT
    # ==========================================

    det_A = np.prod(np.diag(A))
    
    # ==========================================
    # 5. REZOLVAREA SISTEMULUI (Substitutii)
    # ==========================================

    # a) L * z = b 
    z = np.zeros(n)
    for i in range(n):
        sum_lz = sum(A[i, j] * z[j] for j in range(i))
        z[i] = b[i] - sum_lz
        
    # b) D * y = z 
    y = np.zeros(n)
    for i in range(n):
        y[i] = z[i] / A[i][i]
        
    # c) L^T * x = y 
    x_chol = np.zeros(n)
    for i in range(n - 1, -1, -1):
        # A[j, i] in loc de A[i, j] deoarece folosim L transpus
        sum_ltx = sum(A[j, i] * x_chol[j] for j in range(i + 1, n))
        x_chol[i] = y[i] - sum_ltx

    # ==========================================
    # 6. VERIFICAREA SOLUTIEI
    # ==========================================
    
    Ax_calc = np.zeros(n)
    for i in range(n):
        sum_ax = 0
        for j in range(n):
            if j > i:
                # Elementele strict deasupra diagonalei (intacte)
                sum_ax += A[i, j] * x_chol[j]
            elif j == i:
                # Diagonala! Aici NU folosim A[i, i] pentru ca acolo e D acum.
                # Folosim vectorul salvat initial.
                sum_ax += diag_A_original[i] * x_chol[i]
            else:
                # Elementele de sub diagonala (folosim simetria din partea de sus)
                sum_ax += A[j, i] * x_chol[j] 
        Ax_calc[i] = sum_ax

    norm1 = np.linalg.norm(Ax_calc - b, ord=2)
    norm2 = np.linalg.norm(x_chol - x_lib, ord=2)
    
    print(f"Determinantul calculat (det D): {det_A}")
    print(f"Norma 1: ||A_init * x_chol - b||_2 = {norm1}")
    print(f"Norma 2: ||x_chol - x_lib||_2       = {norm2}")
    if norm1 < 1e-8 and norm2 < 1e-8:
        print("-> REZULTAT CORECT: Normele sunt mai mici decat 10^-8!")
    else:
        print("-> AVERTISMENT: Normele nu indeplinesc criteriul de precizie optima.")

# Apelam pentru n mai mare de 100 si eps mai mic de 1e-8
solve_tema2(n=105, eps=1e-9)