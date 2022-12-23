import numpy as np
import matplotlib.pyplot as plt

# def zlatni_presek(f, a, b, max_error = 0.01):
#     c = (3 - np.sqrt(5))/2

#     x1 = a + c * (b - a)
#     x2 = a + b - x1

#     steps = 1
#     while b - a > max_error:
#         if f(x1) < f(x2):
#             b = x2
#             x1 = a + c * (b - a)
#             x2 = a + b - x1
#         else:
#             a = x1
#             x1 = a + c * (b - a)
#             x2 = a + b - x1
#         steps += 1

#     if f(x1) < f(x2):
#         x = x1
#     else:
#         x = x2

#     return x, f(x), steps

def zlatni_presek(f, a, b, max_error = 0.01):
    c = (3 - np.sqrt(5))/2

    x1 = a + c * (b - a)
    x2 = a + b - x1

    steps = 1
    while b - a > max_error:
        if f(x1) < f(x2):
            b = x2
            x1 = a + c * (b - a)
            x2 = a + b - x1
        else:
            a = x1
            x1 = a + c * (b - a)
            x2 = a + b - x1
        steps += 1

    if f(x1) < f(x2):
        x = x1
    else:
        x = x2
    
    return x, f(x), steps

def main():
    f = lambda x: -(x**4 - 5*x**3 - 2*x**2 + 24*x)

    a = 0
    b = 3

    x = np.linspace(a, b, 1000)

    xopt, fopt, steps = zlatni_presek(f, a, b)

    plt.plot(x, f(x))
    plt.scatter(xopt, fopt)
    plt.show()


if __name__ == "__main__":
    main()