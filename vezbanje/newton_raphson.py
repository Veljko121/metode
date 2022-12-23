import numpy as np
import matplotlib.pyplot as plt

def newton_raphson(f, df, ddf, x0, max_error = 0.01, max_steps = 1000):
    for steps in range(max_steps + 1):
        x = x0 - df(x0)/ddf(x0)
        if abs(x - x0) < max_error:
            break
        x0 = x
    return x, steps

def main():
    f = lambda x: (x - 5)**2 + 3
    df = lambda x: 2*x - 10
    ddf = lambda x: 2

    x = np.linspace(0, 10, 1000)

    xopt, steps = newton_raphson(f, df, ddf, 0)

    fopt = f(xopt)

    print(f"x = {xopt}")
    print(f"f(x) = {fopt}")
    print(f"steps: {steps}")

    plt.plot(x, f(x))
    plt.scatter(xopt, fopt)
    plt.show()


if __name__ == "__main__":
    main()