import numpy as np
import matplotlib.pyplot as plt


def steepest_descent(gradf, x0, gamma = 0.1, max_error = 0.01, max_steps = 1000):
    x = np.array(x0).reshape(len(x0), 1)
    for steps in range(max_steps):
        g = gradf(x)
        x = x - gamma*g
        if np.linalg.norm(g) < max_error:
            break
    return x, steps

def steepest_descent_with_momentum(gradf, x0, gamma = 0.1, omega = 0.1, max_error = 0.01, max_steps = 1000):
    x = np.array(x0).reshape(len(x0), 1)
    v = np.zeros(shape=x.shape)
    for steps in range(max_steps):
        g = gradf(x)
        v = omega * v + gamma * g
        x = x - v
        if np.linalg.norm(g) < max_error:
            break
    return x, steps

def nesterov_gradient_descent(gradf, x0, gamma = 0.1, omega = 0.1, max_error = 0.01, max_steps = 1000):
    x = np.array(x0).reshape(len(x0), 1)
    v = np.zeros(shape=x.shape)
    for steps in range(max_steps):
        g = gradf(x - omega * v)
        v = omega * v + gamma * g
        x = x - v
        if np.linalg.norm(g) < max_error:
            break
    return x, steps

def adagrad(gradf, x0, gamma = 0.1, epsilon1 = 0.1, max_error = 0.1, max_steps = 1000):
    x = np.array(x0).reshape(len(x0), 1)
    v = np.zeros(shape=x.shape)
    G = np.zeros(shape=x.shape)
    for steps in range(max_steps):
        g = np.asarray(gradf(x))
        G = G + np.multiply(g, g)
        v = gamma * np.ones(shape=G.shape)/np.sqrt(G + epsilon1) * g
        x = x - v
        if np.linalg.norm(g) < max_error:
            break
    return x, G, steps

def adam(gradf, x0, gamma = 0.1, omega1 = 0.1, omega2 = 0.1, epsilon1 = 0.1, max_error = 0.01, max_steps = 1000):
    x = np.array(x0).reshape(len(x0), 1)
    v = np.ones(shape=x.shape)
    m = np.ones(shape=x.shape)
    for steps in range(max_steps):
        g = np.asarray(gradf(x))
        m = omega1 * m + (1 - omega1) * g
        v = omega2 * v + (1 - omega2) * np.multiply(g, g)
        hat_v = np.abs(v/(1 - omega2))
        hat_m = m/(1 - omega1)
        x = x - gamma * np.ones(shape=g.shape)/np.sqrt(hat_v + epsilon1) * hat_m
        if np.linalg.norm(g) < max_error:
            break
    return x, v, m, steps

# PRIMER UPOTREBE

# f = lambda x: x[0]**2 + x[1]**2
# gradf = lambda x: np.array([2*x[0], 2*x[1]])

# def main():
#     x0 = np.array([2, 5])

#     x, steps = steepest_descent(df, x0)
#     # x, steps = steepest_descent_with_momentum(df, x0)
#     # x, steps = nesterov_gradient_descent(df, x0)
#     # x, steps = adagrad(df, [2, 5], 0.1, 0.1, 0.01, 1000)
#     # x, G, steps = adagrad(df, [2, 5])
#     # x, v, m, steps = adam(df, [2, 5])

#     x1v = np.linspace(-3, 3, 1000)
#     x2v = np.linspace(-3, 3, 1000)

#     x1, x2 = np.meshgrid(x1v, x2v)
#     fig = plt.figure()
#     ax = fig.add_subplot(projection="3d")
#     ax.plot_surface(x1, x2, f([x1, x2]))
#     ax.scatter(x, f(x))
#     plt.show()


# if __name__ == "__main__":
#     main()
