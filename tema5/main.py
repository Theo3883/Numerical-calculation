import numpy as np


# ---------------------------------------------------------------------------
# Tema 5:
# 1) Pentru p=n, aproximarea valorilor/vetorilor proprii ai unei matrice
#    simetrice A folosind metoda Jacobi.
# 2) Verificarea relatiei A_init * U ≈ U * Lambda.
# 3) Construirea sirului A^(k) prin factorizari Cholesky:
#      A^(k) = L^(k)(L^(k))^T, A^(k+1) = (L^(k))^T L^(k).
# 4) Pentru p>n, calcul SVD + rang + conditionare + pseudo-inverse.
# ---------------------------------------------------------------------------


def este_simetrica(a, eps=1e-12):
	# Verificare A = A^T in toleranta numerica eps
	return np.allclose(a, a.T, atol=eps)


def este_diagonala(a, eps=1e-10):
	# Verificare ca toate elementele nediagonale sunt sub pragul eps
	off_diag = a - np.diag(np.diag(a))
	return np.max(np.abs(off_diag)) <= eps


def jacobi_valori_vectori_proprii(a_init, eps=1e-10, k_max=10_000):
	# ==========================================
	# 1. METODA JACOBI PENTRU VALORI PROPRII
	# ==========================================
	# Se construieste sirul A^(k+1) = R_pq * A^(k) * R_pq^T, unde p,q sunt
	# indicii elementului nediagonal maxim in modul.

	a = a_init.astype(float).copy()
	n = a.shape[0]
	# U acumuleaza rotatiile si va contine vectorii proprii aproximativi
	u = np.eye(n)

	def calculeaza_indici_pq(m):
		# Cautam elementul nediagonal de modul maxim in triunghiul inferior
		p_local, q_local = 0, 1
		max_offdiag_local = 0.0
		for i in range(1, n):
			for j in range(i):
				val = abs(m[i, j])
				if val > max_offdiag_local:
					max_offdiag_local = val
					p_local, q_local = i, j
		return p_local, q_local

	def calculeaza_c_s_t(m, p_local, q_local):
		a_pp = m[p_local, p_local]
		a_qq = m[q_local, q_local]
		a_pq = m[p_local, q_local]
		if abs(a_pq) <= eps:
			# Cand a_pq este deja aproape zero, nu aplicam rotatie.
			return 1.0, 0.0, 0.0

		alpha = (a_pp - a_qq) / (2.0 * a_pq)
		semn_alpha = 1.0 if alpha >= 0 else -1.0
		t_local = -alpha + semn_alpha * np.sqrt(alpha * alpha + 1.0)
		c_local = 1.0 / np.sqrt(1.0 + t_local * t_local)
		s_local = t_local * c_local
		return c_local, s_local, t_local

	# k = 0; U = I_n; calculam initial p, q, respectiv c, s, t
	k = 0
	p, q = calculeaza_indici_pq(a)
	c, s, _ = calculeaza_c_s_t(a, p, q)

	# while (A != matrice diagonala si k <= k_max)
	while (not este_diagonala(a, eps=eps)) and (k <= k_max):
		# Construim explicit matricea de rotatie R_pq(theta)
		r = np.eye(n)
		r[p, p] = c
		r[q, q] = c
		r[p, q] = s
		r[q, p] = -s

		# A = R_pq(theta) * A * R_pq^T(theta)
		a = r @ a @ r.T
		# Simetrizam numeric pentru a elimina erorile de rotunjire
		a = 0.5 * (a + a.T)
		# In teorie, elementul (p, q) se anuleaza exact la fiecare rotatie Jacobi.
		a[p, q] = 0.0
		a[q, p] = 0.0

		# U = U * R_pq^T(theta)
		u = u @ r.T

		# Recalculam p, q, apoi c, s, t pentru urmatorul pas
		p, q = calculeaza_indici_pq(a)
		c, s, _ = calculeaza_c_s_t(a, p, q)

		k += 1

	# La final, valorile proprii aproximative sunt pe diagonala lui A
	lambda_aprox = np.diag(a).copy()
	return lambda_aprox, u, a, k


def verifica_relatie_jacobi(a_init, u, valori_proprii):
	# ==========================================
	# 2. VERIFICAREA A_init * U ≈ U * Lambda
	# ==========================================
	lambda_mat = np.diag(valori_proprii)
	return np.linalg.norm(a_init @ u - u @ lambda_mat)


def sir_matrici_cholesky(a_init, eps=1e-10, k_max=1_000):
	# ==========================================
	# 3. SIRUL A^(k) FOLOSIND FACTORIZARE CHOLESKY
	# ==========================================
	# A^(k) = L^(k)(L^(k))^T
	# A^(k+1) = (L^(k))^T L^(k)

	a_k = a_init.astype(float).copy()

	for k in range(1, k_max + 1):
		try:
			# np.linalg.cholesky returneaza L inferior
			l_k = np.linalg.cholesky(a_k)
		except np.linalg.LinAlgError:
			# Cholesky exista doar pentru matrice simetrice pozitiv definite
			return a_k, k - 1, False, "Matricea nu este pozitiv definita (Cholesky nu poate fi aplicat)."

		# Construim urmatorul termen din sir conform enuntului
		a_next = l_k.T @ l_k
		diferenta = np.linalg.norm(a_next - a_k)
		a_k = a_next

		# Conditie de oprire: ||A^(k)-A^(k-1)|| < eps
		if diferenta < eps:
			return a_k, k, True, f"Convergenta atinsa: ||A^(k)-A^(k-1)|| = {diferenta:.3e}"

	# Oprire fortata la k_max
	return a_k, k_max, False, "S-a atins k_max fara convergenta."


def analiza_svd_p_greater_n(a, eps=1e-12):
	# ==========================================
	# 4. CAZUL p > n: SVD + RANG + CONDITIONARE + PSEUDOINVERSE
	# ==========================================
	# A = U * S * V^T

	p, n = a.shape
	if p <= n:
		raise ValueError("Analiza SVD ceruta aici este pentru cazul p > n.")

	# Descompunerea dupa valori singulare
	u, singular_values, vt = np.linalg.svd(a, full_matrices=True)
	v = vt.T

	# Rang(A): numarul de valori singulare strict pozitive (in toleranta)
	rang_formula = int(np.sum(singular_values > eps))
	# Rang(A) din functie de biblioteca
	rang_biblioteca = int(np.linalg.matrix_rank(a))

	# Numarul de conditionare: sigma_max / sigma_min(strict pozitiva)
	valori_pozitive = singular_values[singular_values > eps]
	if valori_pozitive.size == 0:
		cond_formula = np.inf
	else:
		cond_formula = float(np.max(valori_pozitive) / np.min(valori_pozitive))
	# Comparatie cu functia de biblioteca
	cond_biblioteca = float(np.linalg.cond(a))

	# Pseudoinversa Moore-Penrose A^I
	a_i = np.linalg.pinv(a, rcond=eps)

	# Pseudoinversa in sens MMCP: A^J = (A^T A)^(-1) A^T (daca A^T A este inversabila)
	a_t_a = a.T @ a
	if np.linalg.matrix_rank(a_t_a) == n:
		a_j = np.linalg.inv(a_t_a) @ a.T
		# Norma ceruta in enunt: ||A^I - A^J||_1
		norma_1 = np.linalg.norm(a_i - a_j, ord=1)
	else:
		a_j = None
		norma_1 = None

	return {
		"u": u,
		"s": singular_values,
		"v": v,
		"rang_formula": rang_formula,
		"rang_biblioteca": rang_biblioteca,
		"cond_formula": cond_formula,
		"cond_biblioteca": cond_biblioteca,
		"a_i": a_i,
		"a_j": a_j,
		"norma_1": norma_1,
	}


def ruleaza_exemplu_jacobi(nume, a, valori_asteptate, eps=1e-10):
	# ==========================================
	# 5. RULARE EXEMPLE p=n DIN ENUNT
	# ==========================================
	print(f"\n=== {nume} ===")
	print("A =")
	print(a)

	if not este_simetrica(a):
		print("Matricea nu este simetrica; metoda Jacobi nu se aplica.")
		return

	# Pas 1: Jacobi
	lambda_aprox, u, a_final, iteratii = jacobi_valori_vectori_proprii(a, eps=eps)
	# Sortam valorile proprii pentru comparatie usoara cu valorile din enunt
	idx = np.argsort(lambda_aprox)
	lambda_sortat = lambda_aprox[idx]
	u_sortat = u[:, idx]

	# Pas 2: verificarea relatiei A_init * U ≈ U * Lambda
	norma_verificare = verifica_relatie_jacobi(a, u_sortat, lambda_sortat)

	print(f"Iteratii Jacobi: {iteratii}")
	print("Valori proprii aproximative (sortate):")
	print(lambda_sortat)
	print("Valori proprii asteptate (enunt, sortate):")
	print(np.sort(np.array(valori_asteptate, dtype=float)))
	print("Matricea U (coloane = vectori proprii aproximativi):")
	print(u_sortat)
	print("A_final (aproximativ diagonala):")
	print(a_final)
	print(f"||A_init * U - U * Lambda|| = {norma_verificare:.3e}")

	# Pas 3: sirul de matrici bazat pe factorizare Cholesky
	a_limita, k_chol, convergent, mesaj = sir_matrici_cholesky(a, eps=eps)
	print("\nSirul A^(k) prin factorizare Cholesky:")
	print(f"Iteratii: {k_chol}, convergenta: {convergent}")
	print(mesaj)
	print("Ultima matrice calculata:")
	print(a_limita)


def ruleaza_exemplu_svd(nume, a_rect, eps=1e-12):
	# ==========================================
	# 6. RULARE EXEMPLE p>n
	# ==========================================
	print(f"\n=== {nume} ===")
	print("A (p>n) =")
	print(a_rect)

	rezultate = analiza_svd_p_greater_n(a_rect, eps=eps)

	print("Valorile singulare ale lui A:")
	print(rezultate["s"])
	print(f"Rang(A) din formula (sigma_i > eps): {rezultate['rang_formula']}")
	print(f"Rang(A) din biblioteca: {rezultate['rang_biblioteca']}")
	print(f"k2(A) din formula: {rezultate['cond_formula']}")
	print(f"k2(A) din biblioteca: {rezultate['cond_biblioteca']}")

	print("Pseudoinversa Moore-Penrose A^I:")
	print(rezultate["a_i"])

	if rezultate["a_j"] is not None:
		print("Pseudoinversa in sens MMCP A^J = (A^T A)^(-1) A^T:")
		print(rezultate["a_j"])
		print(f"||A^I - A^J||_1 = {rezultate['norma_1']:.3e}")
	else:
		# Daca A^T A e singulara, formula clasica pentru A^J nu se poate aplica
		print("A^T A nu este inversabila; A^J nu poate fi calculata cu formula (A^T A)^(-1)A^T.")


if __name__ == "__main__":
	# ==========================================
	# 7. DATE DE TEST
	# ==========================================
	np.set_printoptions(precision=6, suppress=True)

	exemple_simetrice = [
		(
			"Exemplul 1",
			np.array([
				[0.0, 0.0, 1.0],
				[0.0, 0.0, 1.0],
				[1.0, 1.0, 1.0],
			]),
			[-1.0, 0.0, 2.0],
		),
		(
			"Exemplul 2",
			np.array([
				[1.0, 1.0, 2.0],
				[1.0, 1.0, 2.0],
				[2.0, 2.0, 2.0],
			]),
			[2.0 * (1.0 - np.sqrt(2.0)), 0.0, 2.0 * (1.0 + np.sqrt(2.0))],
		),
		(
			"Exemplul 3",
			np.array([
				[1.0, 0.0, 1.0, 0.0],
				[0.0, 1.0, 0.0, 1.0],
				[1.0, 0.0, 1.0, 0.0],
				[0.0, 1.0, 0.0, 1.0],
			]),
			[0.0, 0.0, 2.0, 2.0],
		),
		(
			"Exemplul 4",
			np.array([
				[1.0, 2.0, 3.0, 4.0],
				[2.0, 3.0, 4.0, 5.0],
				[3.0, 4.0, 5.0, 6.0],
				[4.0, 5.0, 6.0, 7.0],
			]),
			[0.0, 0.0, 2.0 * (4.0 - np.sqrt(21.0)), 2.0 * (4.0 + np.sqrt(21.0))],
		),
	]

	# Rulare pe matricile simetrice p=n din enunt
	for nume, a, val_asteptate in exemple_simetrice:
		ruleaza_exemplu_jacobi(nume, a, val_asteptate, eps=1e-10)

	# Exemplu p>n 
	exemplu_p_mai_mare_n = np.array([
		[0.0, 0.0, 1.0],
		[1.0, 1.0, 1.0],
		[1.0, 2.0, 3.0],
		[2.0, 3.0, 4.0],
	])
	# Rulare pentru cerintele de la punctul p>n
	ruleaza_exemplu_svd(
		"Cazul p > n (exemplu derivat din valorile din enunt)",
		exemplu_p_mai_mare_n,
		eps=1e-12,
	)
