from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def horner_eval(coeffs, x_value):
	"""Evalueaza polinomul in x_value cu schema Horner.

	Coeficientii sunt in ordine crescatoare: [a0, a1, ..., am].
	"""
	# Pornim de la coeficientul gradului maxim si coboram spre a0.
	result = coeffs[-1]
	for coeff in reversed(coeffs[:-1]):
		result = result * x_value + coeff
	return result


def least_squares_poly_coeffs(x_nodes, y_nodes, m):
	"""Calculeaza coeficientii polinomului P_m prin CMMP.

	Implementam exact sistemul normal B * a = f din enunt.
	"""
	n_points = len(x_nodes)
	if m >= n_points:
		raise ValueError("Gradul m trebuie sa fie mai mic decat numarul de puncte.")

	# ==========================================
	# 1. CONSTRUIRE SISTEM NORMAL B * a = f
	# ==========================================
	b_mat = np.zeros((m + 1, m + 1), dtype=float)
	f_vec = np.zeros(m + 1, dtype=float)

	for i in range(m + 1):
		for j in range(m + 1):
			b_mat[i, j] = np.sum(x_nodes ** (i + j))
		f_vec[i] = np.sum(y_nodes * (x_nodes ** i))

	# ==========================================
	# 2. REZOLVARE SISTEM LINIAR
	# ==========================================
	return np.linalg.solve(b_mat, f_vec)


def clamped_c2_spline_second_derivatives(x_nodes, y_nodes, d_a, d_b):
	"""Calculeaza vectorul A pentru spline cubic C^2 cu capete fixate."""
	n = len(x_nodes) - 1
	h = np.diff(x_nodes)

	# ==========================================
	# 1. CONSTRUIRE SISTEM H * A = f
	# ==========================================
	h_mat = np.zeros((n + 1, n + 1), dtype=float)
	f_vec = np.zeros(n + 1, dtype=float)

	# Prima ecuatie: foloseste derivata cunoscuta in a.
	h_mat[0, 0] = 2.0 * h[0]
	h_mat[0, 1] = h[0]
	f_vec[0] = 6.0 * ((y_nodes[1] - y_nodes[0]) / h[0] - d_a)

	# Ecuatiile interioare i = 1..n-1.
	for i in range(1, n):
		h_mat[i, i - 1] = h[i - 1]
		h_mat[i, i] = 2.0 * (h[i - 1] + h[i])
		h_mat[i, i + 1] = h[i]
		f_vec[i] = 6.0 * (
			(y_nodes[i + 1] - y_nodes[i]) / h[i]
			- (y_nodes[i] - y_nodes[i - 1]) / h[i - 1]
		)

	# Ultima ecuatie: foloseste derivata cunoscuta in b.
	h_mat[n, n - 1] = h[n - 1]
	h_mat[n, n] = 2.0 * h[n - 1]
	f_vec[n] = 6.0 * (d_b - (y_nodes[n] - y_nodes[n - 1]) / h[n - 1])

	# ==========================================
	# 2. REZOLVARE SISTEM LINIAR
	# ==========================================
	return np.linalg.solve(h_mat, f_vec)


def spline_eval(x_nodes, y_nodes, a_second, x_value):
	#Evalueaza spline-ul in punctul x_value pe intervalul potrivit."""
	# Gasim intervalul [x_i, x_{i+1}] care contine punctul cerut.
	i = np.searchsorted(x_nodes, x_value) - 1
	i = int(np.clip(i, 0, len(x_nodes) - 2))

	x_i = x_nodes[i]
	x_ip1 = x_nodes[i + 1]
	y_i = y_nodes[i]
	y_ip1 = y_nodes[i + 1]
	a_i = a_second[i]
	a_ip1 = a_second[i + 1]
	h_i = x_ip1 - x_i

	# Coeficientii b_i si c_i din formula spline-ului din enunt.
	b_i = (y_ip1 - y_i) / h_i - h_i * (a_ip1 - a_i) / 6.0
	c_i = (x_ip1 * y_i - x_i * y_ip1) / h_i - h_i * (x_ip1 * a_i - x_i * a_ip1) / 6.0

	return (
		((x_value - x_i) ** 3) * a_ip1 / (6.0 * h_i)
		+ ((x_ip1 - x_value) ** 3) * a_i / (6.0 * h_i)
		+ b_i * x_value
		+ c_i
	)


def generate_nodes(a, b, n, func, seed):
	"""Genereaza x_0=a < ... < x_n=b si calculeaza y_i = f(x_i)."""
	rng = np.random.default_rng(seed)
	inner = np.sort(rng.uniform(a, b, size=n - 1))
	x_nodes = np.concatenate(([a], inner, [b]))
	y_nodes = func(x_nodes)
	return x_nodes, y_nodes


def run_example(example_index, example_data, n=10, m=5, output_dir=None):
	# ==========================================
	# 1. EXTRAGERE DATE DE INTRARE
	# ==========================================
	func = example_data["f"]
	d_func = example_data["df"]
	a = example_data["a"]
	b = example_data["b"]
	x_bar = example_data["x_bar"]

	# ==========================================
	# 2. GENERARE NODURI x_i SI VALORI y_i
	# ==========================================
	x_nodes, y_nodes = generate_nodes(a, b, n, func, seed=2026 + example_index)

	# Daca x_bar coincide cu un nod (caz rar), il deplasam foarte putin.
	if np.any(np.isclose(x_nodes, x_bar)):
		x_bar += 1e-6

	# ==========================================
	# 3. APROXIMARE CMMP + HORNER (BAREM)
	# ==========================================
	poly_coeffs = least_squares_poly_coeffs(x_nodes, y_nodes, m)

	# Evaluarea in x_bar se face explicit cu schema Horner.
	p_x_bar = horner_eval(poly_coeffs, x_bar)

	# Calculam si valorile in noduri pentru suma erorilor ceruta in enunt.
	p_node_values = np.array([horner_eval(poly_coeffs, value) for value in x_nodes])
	poly_abs_error = abs(p_x_bar - func(x_bar))
	poly_sum_abs = np.sum(np.abs(p_node_values - y_nodes))

	# ==========================================
	# 4. APROXIMARE SPLINE C^2 (BAREM)
	# ==========================================
	a_second = clamped_c2_spline_second_derivatives(
		x_nodes,
		y_nodes,
		d_func(a),
		d_func(b),
	)
	s_x_bar = spline_eval(x_nodes, y_nodes, a_second, x_bar)
	spline_abs_error = abs(s_x_bar - func(x_bar))

	# ==========================================
	# 5. AFISARE REZULTATE CERUTE
	# ==========================================
	print(f"\nExemplul {example_index + 1}: {example_data['name']}")
	print(f"Interval: [{a}, {b}], n = {n}, m = {m}, x_bar = {x_bar}")
	print(f"|P_m(x_bar) - f(x_bar)| = {poly_abs_error:.12g}")
	print(f"sum_i |P_m(x_i) - y_i| = {poly_sum_abs:.12g}")
	print(f"|S_f(x_bar) - f(x_bar)| = {spline_abs_error:.12g}")

	# ==========================================
	# 6. GRAFIC f, P_m, S_f (BAREM)
	# ==========================================
	
	x_dense = np.linspace(a, b, 500)
	y_true = func(x_dense)
	y_poly = np.array([horner_eval(poly_coeffs, value) for value in x_dense])
	y_spline = np.array([spline_eval(x_nodes, y_nodes, a_second, value) for value in x_dense])

	plt.figure(figsize=(10, 6))
	plt.plot(x_dense, y_true, label="f(x)", linewidth=2.0)
	plt.plot(x_dense, y_poly, "--", label=f"P_{m}(x) CMMP", linewidth=1.8)
	plt.plot(x_dense, y_spline, "-.", label="S_f(x) spline C^2", linewidth=1.8)
	plt.scatter(x_nodes, y_nodes, color="black", s=24, label="Noduri")
	plt.scatter([x_bar], [func(x_bar)], color="green", marker="o", s=50, label="f(x_bar)")
	plt.scatter([x_bar], [p_x_bar], color="tab:orange", marker="x", s=60, label="P_m(x_bar)")
	plt.scatter([x_bar], [s_x_bar], color="tab:red", marker="+", s=70, label="S_f(x_bar)")
	plt.title(f"Tema 6 - Aproximari numerice ({example_data['name']})")
	plt.xlabel("x")
	plt.ylabel("valoare")
	plt.grid(alpha=0.3)
	plt.legend()
	plt.tight_layout()

	if output_dir is not None:
		plot_path = output_dir / f"plot_exemplu_{example_index + 1}.png"
		plt.savefig(plot_path, dpi=140)
		print(f"Grafic salvat: {plot_path}")
	plt.close()


def main():
	# ==========================================
	# DATE DE INTRARE - EXEMPLE DIN ENUNT
	# ==========================================
	examples = [
		{
			"name": "f(x)=x^4-12x^3+30x^2+12",
			"a": 0.0,
			"b": 2.0,
			"x_bar": 1.5,
			"f": lambda x: x**4 - 12.0 * x**3 + 30.0 * x**2 + 12.0,
			"df": lambda x: 4.0 * x**3 - 36.0 * x**2 + 60.0 * x,
		},
		{
			"name": "f(x)=x^3+3x^2-5x+12",
			"a": 1.0,
			"b": 5.0,
			"x_bar": 2.5,
			"f": lambda x: x**3 + 3.0 * x**2 - 5.0 * x + 12.0,
			"df": lambda x: 3.0 * x**2 + 6.0 * x - 5.0,
		},
	]

	# ==========================================
	# RULARE PENTRU FIECARE EXEMPLU
	# ==========================================
	base_dir = Path(__file__).parent
	for idx, example in enumerate(examples):
		run_example(idx, example, n=10, m=5, output_dir=base_dir)


if __name__ == "__main__":
	main()
