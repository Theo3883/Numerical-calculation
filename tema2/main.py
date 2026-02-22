import numpy as np
import scipy.linalg

def solve_tema2(n=101, eps=1e-8):
    # ==========================================
    # 1. GENERAREA DATELOR
    # ==========================================
    print(f"Running for n={n} eps={eps}")
    
    # Generam matricea oarecare B si construim A = B * B^T (simetrica si pozitiv definita)
    B = np.random.rand(n, n)
    A = np.dot(B, B.T) 
    
    # Generam vectorul termenilor liberi b
    b = np.random.rand(n)
    
    # Salvam o copie STRICT pentru a verifica la final corectitudinea (nu se va folosi in calcul)
    A_copy_for_validation = A.copy()

    # ==========================================
    # 2. REZOLVARE CU BIBLIOTECA MATEMATICA
    # ==========================================
    # Folosim scipy.linalg.lu_factor si lu_solve pentru descompunerea LU
    lu, piv = scipy.linalg.lu_factor(A)
    x_lib = scipy.linalg.lu_solve((lu, piv), b)

    # ==========================================
    # 3. DESCOMPUNEREA LDL^T (In-place)
    # ==========================================
    d = np.zeros(n) # Aici vom memora diagonala matricei D
    
    for p in range(n):
        # Calculam elementul diagonal d_p
        sum_dp = sum(d[k] * A[p, k]**2 for k in range(p))
        d[p] = A[p, p] - sum_dp
        
        # Inlocuim if(v!=0) cu verificarea abs(v) > eps
        if abs(d[p]) <= eps:
            raise ValueError(f"Impartire la zero prevenita! d[{p}] este prea mic (<= eps).")
            
        # Calculam elementele de pe coloana p a matricei L
        for i in range(p + 1, n):
            sum_lip = sum(d[k] * A[i, k] * A[p, k] for k in range(p))
            A[i, p] = (A[i, p] - sum_lip) / d[p]
            # Elementul proaspat calculat se salveaza direct in A (sub diagonala principala)

    # ==========================================
    # 4. CALCUL DETERMINANT
    # ==========================================
    # det(A) = produs(d_i) deoarece det(L) = 1 si det(L^T) = 1
    det_A = np.prod(d)
    
    # ==========================================
    # 5. REZOLVAREA SISTEMULUI (Substitutii)
    # ==========================================
    # a) L * z = b (Substitutie directa cu 1 pe diagonala)
    z = np.zeros(n)
    for i in range(n):
        sum_lz = sum(A[i, j] * z[j] for j in range(i))
        z[i] = b[i] - sum_lz
        
    # b) D * y = z (Sistem diagonal)
    y = np.zeros(n)
    for i in range(n):
        y[i] = z[i] / d[i]
        
    # c) L^T * x = y (Substitutie inversa cu 1 pe diagonala)
    x_chol = np.zeros(n)
    for i in range(n - 1, -1, -1):
        # Atentie: elementele lui L^T sunt A[j, i] din cauza transpusei
        sum_ltx = sum(A[j, i] * x_chol[j] for j in range(i + 1, n))
        x_chol[i] = y[i] - sum_ltx

    # ==========================================
    # 6. VERIFICAREA SOLUTIEI
    # ==========================================
    # Procedura custom de inmultire A_init * x_chol fara a folosi o copie a lui A
    # Partea strict inferior triunghiulara a fost suprascrisa cu L.
    # Ne folosim de faptul ca A initial a fost simetrica, deci folosim doar A[i, j] cu j >= i
    Ax_calc = np.zeros(n)
    for i in range(n):
        sum_ax = 0
        for j in range(n):
            if j >= i:
                sum_ax += A[i, j] * x_chol[j] # Din partea superior triunghiulara (ramasa intacta)
            else:
                sum_ax += A[j, i] * x_chol[j] # Folosim simetria (a_ij = a_ji)
        Ax_calc[i] = sum_ax

    # Calculam normele cerute
    norm1 = np.linalg.norm(Ax_calc - b, ord=2)
    norm2 = np.linalg.norm(x_chol - x_lib, ord=2)
    
    print(f"Determinantul calculat (det D): {det_A}")
    print(f"Norma 1: ||A_init * x_chol - b||_2 = {norm1}")
    print(f"Norma 2: ||x_chol - x_lib||_2       = {norm2}")
    if norm1 < 1e-8 and norm2 < 1e-8:
        print("-> REZULTAT CORECT: Normele sunt mai mici decat 10^-8!")
    else:
        print("-> AVERTISMENT: Normele nu indeplinesc criteriul de precizie optima.")

# Apelam functia pentru n > 100 cum este precizat in PDF
solve_tema2(n=105, eps=1e-9)