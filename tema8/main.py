from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


Array = np.ndarray


@dataclass
class FunctionSpec:
	# Container pentru functia obiectiv si gradientul ei analitic.
	name: str
	f: Callable[[Array], float]
	grad: Callable[[Array], Array]
	known_min: Array | None = None


@dataclass
class GDResult:
	# Rezultatul unei rulari GD pentru o combinatie: (functie, rata, gradient).
	method_name: str
	gradient_mode: str
	start_point: Array
	x: Array
	fx: float
	iterations: int
	converged: bool
	message: str


def sigmoid(z: float | Array) -> float | Array:
	"""Sigmoid numeric stabil pentru argumente mari in modul."""
	z_array = np.asarray(z, dtype=float)
	z_clipped = np.clip(z_array, -500.0, 500.0)
	return 1.0 / (1.0 + np.exp(-z_clipped))


def g_i(f: Callable[[Array], float], x: Array, i: int, h: float = 1e-5) -> float:
	"""Aproximare de ordin inalt pentru derivata partiala dF/dx_i.

	Gi(x, h) = (-F1 + 8F2 - 8F3 + F4) / (12h)
	"""
	e_i = np.zeros_like(x, dtype=float)
	e_i[i] = 1.0
	# Formula cu 4 evaluari ale functiei in jurul lui x pe directia e_i.
	f1 = f(x + 2.0 * h * e_i)
	f2 = f(x + h * e_i)
	f3 = f(x - h * e_i)
	f4 = f(x - 2.0 * h * e_i)
	return float((-f1 + 8.0 * f2 - 8.0 * f3 + f4) / (12.0 * h))


def g1(f: Callable[[Array], float], x: Array, h: float = 1e-5) -> float:
	"""G1(x, h) ~ dF/dx (prima componenta)."""
	return g_i(f, x, i=0, h=h)


def g2(f: Callable[[Array], float], x: Array, h: float = 1e-5) -> float:
	"""G2(x, h) ~ dF/dy (a doua componenta)."""
	if x.size < 2:
		raise ValueError("G2 necesita cel putin 2 dimensiuni.")
	return g_i(f, x, i=1, h=h)


def approximate_gradient(f: Callable[[Array], float], x: Array, h: float = 1e-5) -> Array:
	# Construim gradientul aproximativ componenta cu componenta.
	return np.array([g_i(f, x, i, h) for i in range(x.size)], dtype=float)


def constant_learning_rate(_: Callable[[Array], float], __: Array, ___: Array, eta: float = 1e-2) -> float:
	return eta


def eta_constant_1e2(f: Callable[[Array], float], x: Array, grad_x: Array) -> float:
	return constant_learning_rate(f, x, grad_x, eta=1e-2)


def backtracking_learning_rate(
	f: Callable[[Array], float],
	x: Array,
	grad_x: Array,
	beta: float = 0.8,
	max_backtracking_steps: int = 8,
) -> float:
	# Incepe cu pas maxim si il reduce pana cand conditia Armijo este satisfacuta.
	eta = 1.0
	p = 1
	grad_sq_norm = float(np.dot(grad_x, grad_x))
	fx = f(x)
	while f(x - eta * grad_x) > fx - (eta / 2.0) * grad_sq_norm and p < max_backtracking_steps:
		eta *= beta
		p += 1
	return eta


def eta_backtracking_beta08(f: Callable[[Array], float], x: Array, grad_x: Array) -> float:
	# p porneste de la 0 pentru a permite pana la 8 reduceri efective ale pasului.
	return backtracking_learning_rate(f, x, grad_x, beta=0.8, max_backtracking_steps=9)


def gradient_descent(
	f_spec: FunctionSpec,
	x0: Array,
	epsilon: float,
	k_max: int,
	learning_rate_policy: Callable[[Callable[[Array], float], Array, Array], float],
	gradient_mode: str,
	h: float = 1e-5,
	divergence_guard: float = 1e10,
) -> GDResult:
	# x porneste din punctul initial; pastram si startul pentru raportare.
	x = x0.astype(float).copy()
	start = x.copy()

	for k in range(k_max + 1):
		# Alegere tip gradient: formula analitica sau formula aproximativa Gi.
		if gradient_mode == "analytic":
			grad_x = f_spec.grad(x)
		elif gradient_mode == "approx":
			grad_x = approximate_gradient(f_spec.f, x, h=h)
		else:
			raise ValueError("gradient_mode trebuie sa fie 'analytic' sau 'approx'.")

		eta = learning_rate_policy(f_spec.f, x, grad_x)
		scaled_grad = eta * np.linalg.norm(grad_x, ord=2)

		# Schema de oprire din cerinta: eta * ||grad|| <= epsilon.
		if scaled_grad <= epsilon:
			return GDResult(
				method_name=learning_rate_policy.__name__,
				gradient_mode=gradient_mode,
				start_point=start,
				x=x,
				fx=float(f_spec.f(x)),
				iterations=k,
				converged=True,
				message="Convergenta: eta * ||grad|| <= epsilon",
			)

		if scaled_grad > divergence_guard:
			return GDResult(
				method_name=learning_rate_policy.__name__,
				gradient_mode=gradient_mode,
				start_point=start,
				x=x,
				fx=float(f_spec.f(x)),
				iterations=k,
				converged=False,
				message="Divergenta: eta * ||grad|| > 1e10",
			)

		x = x - eta * grad_x
		# Pasul GD propriu-zis: x_{k+1} = x_k - eta_k * gradF(x_k).

	return GDResult(
		method_name=learning_rate_policy.__name__,
		gradient_mode=gradient_mode,
		start_point=start,
		x=x,
		fx=float(f_spec.f(x)),
		iterations=k_max,
		converged=False,
		message="Stop: numar maxim de iteratii atins",
	)


def build_function_set() -> list[FunctionSpec]:
	# Cele 5 functii din enunt, fiecare cu gradientul analitic explicit.
	def f1(w: Array) -> float:
		w0, w1 = w
		return float(-np.log(1.0 - sigmoid(w0 - w1)) - np.log(sigmoid(w0 + w1)))

	def grad1(w: Array) -> Array:
		w0, w1 = w
		return np.array(
			[
				sigmoid(w0 - w1) + sigmoid(w0 + w1) - 1.0,
				sigmoid(w0 + w1) - sigmoid(w0 - w1) - 1.0,
			],
			dtype=float,
		)

	def f2(x: Array) -> float:
		x1, x2 = x
		return float(x1**2 + x2**2 - 2.0 * x1 - 4.0 * x2 - 1.0)

	def grad2(x: Array) -> Array:
		x1, x2 = x
		return np.array([2.0 * x1 - 2.0, 2.0 * x2 - 4.0], dtype=float)

	def f3(x: Array) -> float:
		x1, x2 = x
		return float(3.0 * x1**2 - 12.0 * x1 + 2.0 * x2**2 + 16.0 * x2 - 10.0)

	def grad3(x: Array) -> Array:
		x1, x2 = x
		return np.array([6.0 * x1 - 12.0, 4.0 * x2 + 16.0], dtype=float)

	def f4(x: Array) -> float:
		x1, x2 = x
		return float(x1**2 - 4.0 * x1 * x2 + 4.5 * x2**2 - 4.0 * x2 + 3.0)

	def grad4(x: Array) -> Array:
		x1, x2 = x
		return np.array([2.0 * x1 - 4.0 * x2, -4.0 * x1 + 9.0 * x2 - 4.0], dtype=float)

	def f5(x: Array) -> float:
		x1, x2 = x
		return float(x1**2 * x2 - 2.0 * x1 * x2**2 + 3.0 * x1 * x2 + 4.0)

	def grad5(x: Array) -> Array:
		x1, x2 = x
		return np.array([2.0 * x1 * x2 - 2.0 * x2**2 + 3.0 * x2, x1**2 - 4.0 * x1 * x2 + 3.0 * x1], dtype=float)

	return [
		FunctionSpec(name="l(w0,w1) logistic", f=f1, grad=grad1, known_min=None),
		FunctionSpec(name="x1^2 + x2^2 - 2x1 - 4x2 - 1", f=f2, grad=grad2, known_min=np.array([1.0, 2.0])),
		FunctionSpec(name="3x1^2 - 12x1 + 2x2^2 + 16x2 - 10", f=f3, grad=grad3, known_min=np.array([2.0, -4.0])),
		FunctionSpec(name="x1^2 - 4x1x2 + 4.5x2^2 - 4x2 + 3", f=f4, grad=grad4, known_min=np.array([8.0, 4.0])),
		FunctionSpec(name="x1^2x2 - 2x1x2^2 + 3x1x2 + 4", f=f5, grad=grad5, known_min=np.array([-1.0, 0.5])),
	]


def format_vec(vec: Array) -> str:
	return np.array2string(vec, precision=6, suppress_small=False)


def print_g1_g2_check(f_spec: FunctionSpec, point: Array, h: float = 1e-5) -> None:
	# Verificare cerinta: G1/G2 fata de componentele gradientului analitic.
	grad_a = f_spec.grad(point)
	g1_val = g1(f_spec.f, point, h=h)
	g2_val = g2(f_spec.f, point, h=h)
	print(f"  Punct verificare G1/G2: x = {format_vec(point)}")
	print(f"  G1 ~ dF/dx: {g1_val:.10f} | analitic: {grad_a[0]:.10f} | eroare: {abs(g1_val - grad_a[0]):.3e}")
	print(f"  G2 ~ dF/dy: {g2_val:.10f} | analitic: {grad_a[1]:.10f} | eroare: {abs(g2_val - grad_a[1]):.3e}")


def run_all_tests() -> None:
	# Configurare experimente: acelasi epsilon/k_max pentru toate functiile.
	functions = build_function_set()
	rng = np.random.default_rng(2026)

	epsilon = 1e-6
	k_max = 30000
	h = 1e-5

	print("Tema 8 - Minimizare cu metoda gradientului descendent")
	print(f"Setari globale: epsilon={epsilon}, k_max={k_max}, h={h}")
	print("Strategii rata invatare: constanta (eta=1e-2) si backtracking (beta=0.8)")

	for index, f_spec in enumerate(functions, start=1):
		if f_spec.known_min is None:
			# Pentru functia logistica alegem un punct moderat, neparticular.
			x0 = rng.uniform(-2.0, 2.0, size=2)
		else:
			# Pentru functiile cu minim cunoscut pornim din vecinatate pentru comparatii stabile.
			x0 = f_spec.known_min + rng.uniform(-0.8, 0.8, size=2)
		print("\n" + "=" * 90)
		print(f"Functia {index}: {f_spec.name}")
		print(f"Punct initial aleator x0 = {format_vec(x0)}")

		# Verificam explicit folosirea G1 si G2 in raport cu gradientul analitic.
		print_g1_g2_check(f_spec, x0, h=h)

		policies = [eta_constant_1e2, eta_backtracking_beta08]

		for policy in policies:
			for grad_mode in ("analytic", "approx"):
				# Rulam toate combinatiile cerute in barem:
				# 2 rate de invatare x 2 moduri de calcul al gradientului.
				result = gradient_descent(
					f_spec=f_spec,
					x0=x0,
					epsilon=epsilon,
					k_max=k_max,
					learning_rate_policy=policy,
					gradient_mode=grad_mode,
					h=h,
				)

				print(
					f"  [{result.method_name:>27}] [{result.gradient_mode:^8}] "
					f"iter={result.iterations:6d} | converged={str(result.converged):5s} | "
					f"x*={format_vec(result.x)} | F(x*)={result.fx:.10f}"
				)

				if f_spec.known_min is not None:
					# Metru de verificare: cat de aproape este de minimul furnizat in enunt.
					dist = np.linalg.norm(result.x - f_spec.known_min)
					print(f"    Distanta fata de minimul din enunt {format_vec(f_spec.known_min)}: {dist:.6e}")
				print(f"    Motiv stop: {result.message}")


def main() -> None:
	run_all_tests()


if __name__ == "__main__":
	main()
