import numpy as np
import matplotlib.pyplot as plt

def steepest_descent(gradf, x0, gamma = 0.1, max_error = 0.01, max_steps = 1000):
    x = np.array(x0).reshape(len(x0), 1)
    for steps in range(max_steps + 1):
        g = gradf(x)
        x = x - gamma * g
        if np.linalg.norm(g) < max_error:
            break
    return x, steps

def steepest_descent_with_momentum(gradf, x0, gamma = 0.1, omega = 0.1, max_error = 0.01, max_steps = 1000):
    x = np.array(x0).reshape(len(x0), 1)
    v = np.zeros(shape=x.shape)
    for steps in range(max_steps + 1):
        g = gradf(x)
        v = omega * v + gamma * g
        x = x - v
        if np.linalg.norm(g) < max_error:
            break
    return x, steps

def nesterov_gradient_descent(gradf, x0, gamma = 0.1, omega = 0.1, max_error = 0.01, max_steps = 1000):
    x = np.array(x0).reshape(len(x0), 1)
    v = np.zeros(shape=x.shape)
    for steps in range(max_steps + 1):
        g = gradf(x - omega * v)
        v = omega * v + gamma * g
        x = x - v
        if np.linalg.norm(g) < max_error:
            break
    return x, steps

def nesterov_gradient_descent(gradf, x0, gamma = 0.1, omega = 0.1, max_error = 0.1, max_steps = 1000):
    x = np.array(x0).reshape(len(x0), 1)
    v = np.zeros(shape=x.shape)
    for steps in range(max_steps + 1):
        g = gradf(x - omega * v)
        v = omega * v + gamma * g
        x = x - v
        if np.linalg.norm(g) < max_error:
            break
    return x, steps

def main():
    f = lambda x: x[0]**2 + x[1]**2
    df = lambda x: np.array([2*x[0], 2*x[1]])

    x0 = np.array([2, 5])

    # x, steps = steepest_descent(df, x0)
    # x, steps = steepest_descent_with_momentum(df, x0)
    x, steps = nesterov_gradient_descent(df, x0)
    # x, steps = adagrad_v(df, [2, 5], 0.1, 0.1, 0.01, 1000)
    # x, G, steps = adagrad_v(df, [2, 5])
    # x, v, m, steps = adam_v(df, [2, 5])

    print(f"x = {x}")
    print(f"f(x) = {f(x)}")
    print(f"steps: {steps}")

    x1v = np.linspace(-3, 3, 1000)
    x2v = np.linspace(-3, 3, 1000)

    x1, x2 = np.meshgrid(x1v, x2v)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(x1, x2, f([x1, x2]))
    ax.scatter(x, f(x))
    plt.show()


if __name__ == "__main__":
    main()