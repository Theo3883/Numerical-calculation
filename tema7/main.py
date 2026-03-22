from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

@dataclass
class MethodResult:
	# Rezultatul unei rulari Newton/Olver pornind dintr-un x0.
	# root      = radacina aproximata (daca exista)
	# iterations= numarul de iteratii efectuate
	# converged = True doar daca metoda a indeplinit criteriul |delta_x| < eps
	root: float | None
	iterations: int
	converged: bool


def horner_value(coeffs: list[float], x: float) -> float:
	# Schema Horner pentru:
	# P(x) = a0*x^n + a1*x^(n-1) + ... + an
	# Recurenta: b0=a0, bi=ai + bi-1*x, iar P(x)=bn.
	b = coeffs[0]
	for a_i in coeffs[1:]:
		b = a_i + b * x
	return b


def derivative_coeffs(coeffs: list[float]) -> list[float]:
	# Din coeficientii lui P construim coeficientii lui P'.
	# Daca P are grad n, atunci:
	# P'(x) = (n*a0)*x^(n-1) + ((n-1)*a1)*x^(n-2) + ... + a_{n-1}
	n = len(coeffs) - 1
	return [coeffs[i] * (n - i) for i in range(n)]


def bound_r(coeffs: list[float]) -> float:
	# Formula teoretica pentru intervalul radacinilor reale R.
	a0 = coeffs[0]
	if a0 == 0.0:
		raise ValueError("a0 trebuie sa fie nenul.")
	a = max(abs(c) for c in coeffs[1:]) if len(coeffs) > 1 else 0.0
	return (abs(a0) + a) / abs(a0)


def is_distinct_root(existing: list[float], candidate: float, eps: float) -> bool:
	# Cerinta de distinctie din enunt:
	# v1 si v2 sunt considerate diferite daca |v1 - v2| > eps.
	return all(abs(candidate - root) > eps for root in existing)


def newton_step(x: float, p: float, dp: float, ddp: float) -> float:
	# Pentru Newton:
	# x_{k+1} = x_k - P(x_k) / P'(x_k)
	# deci delta_x = P/P'.
	# ddp nu se foloseste, dar pastram semnatura comuna cu Olver.
	del x
	del ddp
	return p / dp


def olver_step(x: float, p: float, dp: float, ddp: float) -> float:
	# Pentru Olver:
	# x_{k+1} = x_k - ( P/P' + 1/2 * c_k )
	# c_k = [P(x_k)]^2 * P''(x_k) / [P'(x_k)]^3
	del x
	c_k = (p * p * ddp) / (dp * dp * dp)
	return (p / dp) + 0.5 * c_k


def iterate_method(
	coeffs: list[float],
	d1: list[float],
	d2: list[float],
	x0: float,
	eps: float,
	k_max: int,
	delta_fn: Callable[[float, float, float, float], float],
) -> MethodResult:
	# Schema comuna Newton/Olver (dupa pseudo-codul):
	# - pornim cu x = x0 si k = 0
	# - daca |P'(x_k)| <= eps -> EXIT (nu putem imparti)
	# - calculam delta_x prin formula Newton sau Olver
	# - x <- x - delta_x, k <- k+1
	# - stop daca:
	#     a) |delta_x| < eps  (convergenta)
	#     b) k > k_max sau |delta_x| > 1e8 (divergenta / oprire fortata)
	x = x0
	k = 0

	while True:
		p = horner_value(coeffs, x)
		dp = horner_value(d1, x) if d1 else 0.0
		if abs(dp) <= eps:
			return MethodResult(root=None, iterations=k, converged=False)

		ddp = horner_value(d2, x) if d2 else 0.0
		delta = delta_fn(x, p, dp, ddp)
		x = x - delta
		k += 1

		if abs(delta) < eps:
			return MethodResult(root=x, iterations=k, converged=True)
		if k > k_max or abs(delta) > 1e8:
			return MethodResult(root=None, iterations=k, converged=False)


def find_roots_with_method(
	coeffs: list[float],
	start_points: list[float],
	eps: float,
	k_max: int,
	method_name: str,
) -> tuple[list[float], list[int]]:
	# Pregatim coeficientii pentru derivata 1 si 2 o singura data.
	d1 = derivative_coeffs(coeffs)
	d2 = derivative_coeffs(d1) if d1 else []

	if method_name == "newton":
		delta_fn = newton_step
	elif method_name == "olver":
		delta_fn = olver_step
	else:
		raise ValueError("Metoda necunoscuta. Foloseste 'newton' sau 'olver'.")

	roots: list[float] = []
	steps: list[int] = []

	# Rulam metoda pentru fiecare punct initial x0.
	for x0 in start_points:
		result = iterate_method(coeffs, d1, d2, x0, eps, k_max, delta_fn)
		if result.converged and result.root is not None:
			# Adaugam doar radacini distincte
			if is_distinct_root(roots, result.root, eps):
				roots.append(result.root)
				steps.append(result.iterations)

	# Sortam radacinile pentru afisare stabila, impreuna cu numarul de pasi.
	roots_steps = sorted(zip(roots, steps), key=lambda rs: rs[0])
	if not roots_steps:
		return [], []
	roots_sorted, steps_sorted = zip(*roots_steps)
	return list(roots_sorted), list(steps_sorted)


def generate_start_points(r: float, count: int) -> list[float]:
	# Construim puncte initiale echidistante in [-R, R].
	# Cu cat avem mai multe puncte, cu atat cresc sansele sa gasim
	# mai multe bazine de convergenta ale radacinilor reale.
	if count < 2:
		return [0.0]
	step = (2.0 * r) / (count - 1)
	return [-r + i * step for i in range(count)]


def format_roots(roots: list[float], steps: list[int], eps: float) -> str:
	# Formatare de afisare in consola pentru setul de radacini gasite.
	lines: list[str] = []
	if not roots:
		return "Nu s-au gasit radacini reale distincte."
	for i, (root, k) in enumerate(zip(roots, steps), start=1):
		value = 0.0 if abs(root) < eps else root
		lines.append(f"  r{i} = {value:.12f}, pasi = {k}")
	return "\n".join(lines)


def run_example(
	name: str, coeffs: list[float], eps: float, k_max: int, starts: int
) -> tuple[str, str]:
	# 1) Calculam intervalul teoretic al radacinilor reale.
	r = bound_r(coeffs)
	# 2) Generam puncte initiale in acel interval.
	x0_points = generate_start_points(r, starts)

	# 3) Cautam radacinile cu Newton si Olver.
	n_roots, n_steps = find_roots_with_method(coeffs, x0_points, eps, k_max, "newton")
	o_roots, o_steps = find_roots_with_method(coeffs, x0_points, eps, k_max, "olver")

	# 4) Comparam metodele pe media numarului de iteratii.
	avg_newton = (sum(n_steps) / len(n_steps)) if n_steps else float("inf")
	avg_olver = (sum(o_steps) / len(o_steps)) if o_steps else float("inf")
	faster = "Olver" if avg_olver < avg_newton else "Newton"

	lines = [
		f"=== {name} ===",
		f"Coeficienti: {coeffs}",
		f"Interval radacini reale (teorie): [-R, R], R = {r:.12f}",
		"Newton - radacini reale distincte:",
		format_roots(n_roots, n_steps, eps),
		"Olver - radacini reale distincte:",
		format_roots(o_roots, o_steps, eps),
	]

	if n_steps and o_steps:
		lines.extend(
			[
				f"Media pasi Newton: {avg_newton:.4f}",
				f"Media pasi Olver:  {avg_olver:.4f}",
				f"Metoda mai rapida (medie): {faster}",
			]
		)
	else:
		lines.append("Comparatie indisponibila: una dintre metode nu a gasit radacini.")

	# In fisier scriem doar radacinile distincte, conform cerintei.
	# Distinctia este deja aplicata in find_roots_with_method, folosind pragul eps.
	roots_only_lines = [
		f"[{name}]",
		"Newton:",
		" ".join(f"{r:.12f}" for r in n_roots) if n_roots else "-",
		"Olver:",
		" ".join(f"{r:.12f}" for r in o_roots) if o_roots else "-",
	]

	return "\n".join(lines), "\n".join(roots_only_lines)


def main() -> None:
	# eps controleaza atat criteriul de oprire, cat si distinctia radacinilor.
	# Alegem 1e-6 pentru a evita clasificarea artificiala a aceleiasi radacini
	# multiple in zeci de valori foarte apropiate numeric.
	eps = 1e-6
	k_max = 1_000
	starts = 250

	# Exemplele din enunt
	examples: list[tuple[str, list[float]]] = [
		("Exemplul 1", [1.0, -6.0, 11.0, -6.0]),
		("Exemplul 2", [42.0, -55.0, -42.0, 49.0, -6.0]),
		("Exemplul 3", [8.0, -38.0, 49.0, -22.0, 3.0]),
		("Exemplul 4", [1.0, -6.0, 13.0, -12.0, 4.0]),
	]

	# Rulam toate exemplele; afisam raport complet in consola,
	# iar in fisier salvam doar radacinile distincte.
	reports_and_roots = [
		run_example(name, coeffs, eps, k_max, starts) for name, coeffs in examples
	]
	reports = [item[0] for item in reports_and_roots]
	roots_only = [item[1] for item in reports_and_roots]
	full_report = "\n\n".join(reports)

	print(full_report)

	out_path = Path(__file__).parent / "radacini_tema7.txt"
	out_path.write_text("\n\n".join(roots_only) + "\n", encoding="utf-8")
	print(f"\nRadacinile distincte au fost salvate in: {out_path}")


if __name__ == "__main__":
	main()
