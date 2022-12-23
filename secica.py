import numpy as np
import matplotlib.pyplot as plt


def secica(df, x0, x1, max_error = 0.01, max_steps = 1000):
    for steps in range(max_steps):
        x = x0 - df(x0)*(x0 - x1)/(df(x0) - df(x1))
        if abs(x - x1) < max_error:
            break
        x0 = x1
        x1 = x
    return x, steps

# PRIMER UPOTREBE

# f = lambda x: -(x - 5)**2 + 3
# df = lambda x: -2*x + 10

# def main():
#     x0 = -1
#     x1 = 1
#     tol = 1e-2
#     x_min, steps = secica(f, df, x0, x1, tol)
#     y_min = f(x_min)
#     print(f"x_min = {x_min}")
#     print(f"y_min = {y_min}")
#     print(f"steps = {steps}")
#     x = np.linspace(x_min - 5, x_min + 5)
#     plt.plot(x, f(x))
#     plt.scatter(x_min, y_min)
#     plt.show()


# if __name__ == "__main__":
#     main()