import numpy as np
import matplotlib.pyplot as plt

def secica(f, df, x0, x1, max_error = 0.01, max_steps = 1000):
    for steps in range(max_steps + 1):
        x = x0 - df(x0)*(x0 - x1)/(df(x0) - df(x1))
        if abs(x - x1) < max_error:
            break
        x0 = x1
        x1 = x
    return x, steps

def main():
    f = lambda x: (x - 5)**2 + 3
    df = lambda x: 2*x - 10
    ddf = lambda x: 2

    x = np.linspace(0, 10, 1000)

    xopt, steps = secica(f, df, -5, -2)

    fopt = f(xopt)

    print(f"x = {xopt}")
    print(f"f(x) = {fopt}")
    print(f"steps: {steps}")

    plt.plot(x, f(x))
    plt.scatter(xopt, fopt)
    plt.show()


if __name__ == "__main__":
    main()