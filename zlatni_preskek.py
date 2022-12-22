import numpy as np
import math
import matplotlib.pyplot as plt

def zlatni_presek_metod(f, a, b, max_error = 0.01):
    c = (3 - math.sqrt(5))/2

    x1 = a + c*(b - a)
    x2 = a + b - x1

    steps = 1
    while b - a > max_error:
        if f(x1) <= f(x2):
            b = x2
            x1 = a + c*(b - a)
            x2 = a + b - x1
        else:
            a = x1
            x1 = a + c*(b - a)
            x2 = a + b - x1
        steps += 1

    if f(x1) <= f(x2):
        x = x1
    else:
        x = x2

    return x, f(x), steps

# PRIMER UPOTREBE

# f = lambda x: -(x**4 - 5*x**3 - 2*x**2 + 24*x)

# def main():
#     a = 0
#     b = 3
#     tol = 1e-4
    
#     xopt, fopt, steps = zlatni_presek_metod(f, a, b, tol)

#     x = np.linspace(a, b, 1000)

#     plt.plot(x, f(x))
#     plt.scatter(xopt, fopt)
#     plt.show()


# if __name__ == "__main__":
#     main()