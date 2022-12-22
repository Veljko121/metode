import numpy as np
import matplotlib.pyplot as plt


def newton_raphson(f, df, ddf, x0, max_error = 0.01, max_steps = 1000):
    for steps in range(max_steps):
        x = x0 - df(x0)/ddf(x0)
        if abs(x - x0) < max_error:
            break
        x0 = x
    return x, f(x), steps

# PRIMER UPOTREBE

# f = lambda x: (x - 5)**2 + 3
# df = lambda x: 2*x - 10
# ddf = lambda x: 2

# def main():
#     x0 = 0.5
#     tol = 1e-2
#     x_min, y_min, steps = newton_raphson(f, df, ddf, x0)
#     x = np.linspace(x_min - 5, x_min + 5)
#     print(f"x_min = {x_min}")
#     print(f"y_min = {y_min}")
#     print(f"steps = {steps}")
#     plt.plot(x, f(x))
#     plt.scatter(x_min, y_min)
#     plt.show()


# if __name__ == "__main__":
#     main()