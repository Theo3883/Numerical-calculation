import math
import random
import time


# ---------------------------------------------------------------------------
# Problema 1:
# Sa se gaseasca cel mai mic numar pozitiv u > 0, u = 10^(-m), m in N,
# astfel ca: 1.0 +_c u != 1.0  (precizia masina in baza 10)
# ---------------------------------------------------------------------------
def find_smallest_u_base10():
	# Pornim de la u = 1 si impartim la 10 pana cand 1.0 + u = 1.0 in float
	m = 0
	u = 1.0

	while 1.0 + u != 1.0:
		u /= 10.0
		m += 1

	# u a depasit pragul: solutia corecta e pasul anterior
	m_solution = m - 1
	u_solution = 10.0 ** (-m_solution)

	print(f"m = {m_solution}")
	print(f"u = 10^(-m) = {u_solution}")
	print(f"1.0 +_c u = {1.0 + u_solution}  (diferit de 1.0)")
	print(f"1.0 +_c u/10 = {1.0 + 10.0 ** (-(m_solution + 1))}  (egal cu 1.0)")

	return u_solution


# ---------------------------------------------------------------------------
# Problema 2:
# Operatia +_c este neasociativa:
#   x = 1.0, y = u/10, z = u/10  =>  (x+y)+z != x+(y+z)
# Gasiti x, y, z pentru care *_c este neasociativa:
#   (x *_c y) *_c z != x *_c (y *_c z)
# ---------------------------------------------------------------------------
def check_addition_non_associativity(u):
	# Alegem x=1.0 si y=z=u/10 deoarece:
	# - 1.0 +_c (u/10) = 1.0  (u/10 e prea mic ca sa schimbe 1.0)
	# - dar y+z = 2*(u/10) = u/5, suficient de mare incat 1.0 + u/5 != 1.0
	x = 1.0
	y = u / 10.0
	z = u / 10.0

	left  = (x + y) + z
	right = x + (y + z)

	print(f"x = {x}, y = z = {y}")
	print(f"(x +_c y) +_c z = {left}")
	print(f"x +_c (y +_c z) = {right}")
	print(f"Neasociativa? {left != right}")


def build_multiplication():
	# Alegem x = 2^1001, y = z = 2^-1001 (puteri exacte in binar) astfel:
	# - x * y = 2^0 = 1.0  (exact, fara eroare de rotunjire)
	# - y * z = 2^-2002 -> underflow -> 0.0  in float64
	# Rezulta: (x*y)*z = 1.0 * z = z  != 0 = x*(y*z)
	x = 2.0 ** 1001
	y = 2.0 ** -1001
	z = 2.0 ** -1001

	xy   = x * y
	yz   = y * z
	left  = (x * y) * z
	right = x * (y * z)

	print(f"x = {x}")
	print(f"y = {y}")
	print(f"z = {z}")
	print(f"x *_c y = {xy}")
	print(f"y *_c z = {yz}")
	print(f"(x *_c y) *_c z = {left}")
	print(f"x *_c (y *_c z) = {right}")
	print(f"Neasociativa? {left != right}")


# ---------------------------------------------------------------------------
# Problema 3: Aproximarea functiei tangenta
#   - Metoda 1: fractii continue cu algoritmul lui Lentz modificat
#   - Metoda 2: aproximare polinomiala (serie MacLaurin)
# Ambele metode reduc argumentul in (-pi/2, pi/2) apoi in (-pi/4, pi/4).
# Se compara cu math.tan(x) pe 10.000 valori generate aleatoriu.
# ---------------------------------------------------------------------------
def reduce_to_minus_pi_over_2_pi_over_2(x):
	x_reduced = math.fmod(x, math.pi)
	if x_reduced <= -math.pi / 2:
		x_reduced += math.pi
	elif x_reduced > math.pi / 2:
		x_reduced -= math.pi
	return x_reduced


def is_multiple_of_pi_over_2(x, tol=1e-12):
	k = round((2.0 * x) / math.pi)
	return abs(x - k * (math.pi / 2.0)) <= tol


def my_tan(x, epsilon):
	# Fractia continua pentru tan:
	#   tan(x) = x/(1+) (-x^2)/(3+) (-x^2)/(5+) ...
	# b_0=0, a_1=x, b_1=1; pentru j>=2: a_j=-x^2, b_j=2j-1
	# Algoritm Lentz modificat: mic=1e-12 inlocuieste 0 la numitor
	mic = 1e-12
	x_reduced = reduce_to_minus_pi_over_2_pi_over_2(x)

	# Valorile multiple de pi/2 sunt puncte de discontinuitate
	if is_multiple_of_pi_over_2(x_reduced):
		k = round((2.0 * x_reduced) / math.pi)
		if k % 2 != 0:
			return math.copysign(math.inf, x_reduced)
		return 0.0

	# Antisimetrie: tan(-x) = -tan(x)
	if x_reduced < 0.0:
		return -my_tan(-x_reduced, epsilon)

	# Initializare Lentz: f_0 = b_0 = 0, inlocuit cu mic
	f = mic
	c = f
	d = 0.0
	x2 = x_reduced * x_reduced

	max_iter = 10000
	for j in range(1, max_iter + 1):
		# a_1 = x, b_1 = 1; pentru j>=2: a_j = -x^2, b_j = 2j-1
		if j == 1:
			a_j = x_reduced
			b_j = 1.0
		else:
			a_j = -x2
			b_j = 2.0 * j - 1.0

		# Pasii algoritmului Lentz modificat
		d = b_j + a_j * d
		if d == 0.0:
			d = mic

		c = b_j + a_j / c
		if c == 0.0:
			c = mic

		d = 1.0 / d
		delta = c * d        # Delta_j = C_j * D_j
		f *= delta           # f_j = Delta_j * f_{j-1}

		# Criteriu de oprire: |Delta_j - 1| < epsilon
		if abs(delta - 1.0) < epsilon:
			return f

	return f


def my_tan_poly(x):
	# Aproximare prin serie MacLaurin (truncata la gradul 9):
	#   tan(x) ≈ x + (1/3)x^3 + (2/15)x^5 + (17/315)x^7 + (62/2835)x^9
	# Formula lucreaza bine pentru x in (-pi/4, pi/4).
	# Pentru x in [pi/4, pi/2) se foloseste: tan(x) = 1/tan(pi/2 - x)

	# Coeficientii polinomului (calculati o singura data, declarati local)
	c1 = 0.3333333333333333
	c2 = 0.1333333333333333
	c3 = 0.0539682539682540
	c4 = 0.0218694885361552

	x_reduced = reduce_to_minus_pi_over_2_pi_over_2(x)

	# Valorile multiple de pi/2 sunt puncte de discontinuitate
	if is_multiple_of_pi_over_2(x_reduced):
		k = round((2.0 * x_reduced) / math.pi)
		if k % 2 != 0:
			return math.copysign(math.inf, x_reduced)
		return 0.0

	# Antisimetrie: tan(-x) = -tan(x)
	if x_reduced < 0.0:
		return -my_tan_poly(-x_reduced)

	# Reducere in (-pi/4, pi/4): tan(x) = 1/tan(pi/2 - x)
	if x_reduced > math.pi / 4:
		return 1.0 / my_tan_poly(math.pi / 2 - x_reduced)

	# Calcul puteri (structura din enunt: x_2, x_3, x_4=x_2*x_2, x_6=x_4*x_2)
	x_2 = x_reduced * x_reduced
	x_3 = x_2 * x_reduced
	x_4 = x_2 * x_2
	x_6 = x_4 * x_2

	return x_reduced + x_3 * (c1 + c2 * x_2 + c3 * x_4 + c4 * x_6)


def benchmark_tangent_methods(samples=10000, epsilon=1e-12):
	values = [random.uniform(-math.pi / 2, math.pi / 2) for _ in range(samples)]

	start_cf = time.perf_counter()
	cf_values = [my_tan(x, epsilon) for x in values]
	time_cf = time.perf_counter() - start_cf

	start_poly = time.perf_counter()
	poly_values = [my_tan_poly(x) for x in values]
	time_poly = time.perf_counter() - start_poly

	reference_values = [math.tan(x) for x in values]

	errors_cf = [abs(reference_values[i] - cf_values[i]) for i in range(samples)]
	errors_poly = [abs(reference_values[i] - poly_values[i]) for i in range(samples)]

	print("\n3) Aproximarea functiei tangenta")
	print(f"Numar valori: {samples}")
	print(f"Epsilon (Lentz): {epsilon}")

	print("\nMetoda fractii continue (Lentz modificat):")
	print(f"Timp total: {time_cf:.6f} s")
	print(f"Eroare medie |tan(x)-my_tan(x)|: {sum(errors_cf) / samples:.6e}")
	print(f"Eroare maxima |tan(x)-my_tan(x)|: {max(errors_cf):.6e}")

	print("\nMetoda polinomiala:")
	print(f"Timp total: {time_poly:.6f} s")
	print(f"Eroare medie |tan(x)-my_tan(x)|: {sum(errors_poly) / samples:.6e}")
	print(f"Eroare maxima |tan(x)-my_tan(x)|: {max(errors_poly):.6e}")



if __name__ == "__main__":

	print("1) Precizia masina (baza 10)")
	u = find_smallest_u_base10()

	print("\n2a) Neasociativitatea adunarii")
	check_addition_non_associativity(u)

	print("\n2b) Neasociativitatea inmultirii")
	build_multiplication()

	benchmark_tangent_methods(samples=10000, epsilon=1e-12)
