def find_smallest_u_base10():
	m = 0
	u = 1.0

	while 1.0 + u != 1.0:
		u /= 10.0
		m += 1

	m_solution = m - 1
	u_solution = 10.0 ** (-m_solution)

	print(f"m = {m}")
	print(f"u = 10^(-m) = {u}")
	print(f"1.0 + u = {1.0 + u}")
	print(f"1.0 + 10^(-(m+1)) = {1.0 + 10.0 ** (-(m + 1))}")


def build_multiplication():
	# (x * y) = 1 (exact), (y * z) = 0 (underflow)
	x = 2.0 ** 1001
	y = 2.0 ** -1001
	z = 2.0 ** -1001

	xy = x * y
	yz = y * z
	left = (x * y) * z
	right = x * (y * z)

	print(f"x = {x}")
	print(f"y = {y}")
	print(f"z = {z}")
	print(f"x * y = {xy}")
	print(f"y * z = {yz}")
	print(f"(x * y) * z = {left}")
	print(f"x * (y * z) = {right}")



if __name__ == "__main__":

	print("1) Precizia masina (baza 10)")
	find_smallest_u_base10()

	print("\n2) Neasociativitatea inmultirii ")
	build_multiplication()
