import numpy as np
import matplotlib.pyplot as plt


def secica(df, x0, x1, max_error = 0.01, max_steps = 1000):
    for steps in range(max_steps + 1):
        x = x0 - df(x0) * (x1 - x0)/(df(x1) - df(x0))
        if abs(x - x1) < max_error:
            break
        x0 = x1
        x1 = x
    return x, steps

def main():
    f = lambda x: np.cos(2*x) - np.sin(x + 1)
    df = lambda x: -2*np.sin(2*x) - np.cos(x + 1)

    a = 0.5
    b = 2

    x_range = np.linspace(a, b, 1000)

    x, steps = secica(df, 0.5, 2, max_error=0.1)

    print(f"x = {x}")
    print(f"f(x) = {f(x)}")
    print(f"steps = {steps}")

    plt.plot(x_range, f(x_range))
    plt.plot(x_range, df(x_range))
    plt.scatter(x, f(x))
    plt.show()


if __name__ == "__main__":
    main()