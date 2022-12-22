import numpy as np
import matplotlib.pyplot as plt


def fibonacci(n):
    if n in (1, 2):
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)

def fibonacci_method(f, a, b, max_error = 0.01):
    n = 1
    while abs(b - a)/max_error > fibonacci(n):
        n += 1
    
    x1 = a + fibonacci(n - 2)/fibonacci(n)*(b - a)
    x2 = a + b - x1

    for _ in range(2, n + 1):
        if f(x1) <= f(x2):
            b = x2
            x2 = x1
            x1 = a + b - x2
        else:
            a = x1
            x1 = x2
            x2 = a + b - x1

    if f(x1) < f(x2):
        x = x1
    else:
        x = x2

    return x, f(x), n

# PRIMER UPOTREBE

# f = lambda x: -(x**4 - 5*x**3 - 2*x**2 + 24*x)

# def main():
#     a = 0
#     b = 3
#     tol = 1e-4

#     xopt, fopt, n = fibonacci_method(f, a, b, tol)

#     print(f"xopt = {xopt}\nfopt = {fopt}\nn = {n}")

#     x = np.linspace(a, b, 1000)
#     plt.plot(x, f(x))
#     plt.scatter(xopt, fopt)
#     plt.show()


# if __name__ == "__main__":
#     main()