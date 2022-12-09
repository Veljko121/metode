import numpy as np
import matplotlib.pyplot as plt


def steepest_descent(gradf, x0, gamma=1e-1, epsilon=1e-2, max_steps=1000):
    x = np.array(x0).reshape(len(x0), 1)
    for steps in range(max_steps):
        g = gradf(x)
        x = x - g*gamma
        if np.linalg.norm(g) < epsilon:
            break
    return x
    
def steepest_descent_with_momentum(gradf, x0, gamma=1e-1, omega=1e-1, epsilon=1e-2, max_steps=1000):
    x = np.array(x0).reshape(len(x0), 1)
    v = np.zeros(shape=x.shape)
    for steps in range(max_steps):
        g = gradf(x)
        v = v*omega + g*gamma
        x = x - v
        if np.linalg.norm(g) < epsilon:
            break
    return x

def adam_v(gradf, x0, gamma=1e-1, omega1=1e-1, omega2=1e-1, epsilon1=1e-2, epsilon=1e-2, max_steps=1000):
    x = np.array(x0).reshape(len(x0), 1)
    v = np.zeros(shape=x.shape)
    m = np.ones(shape=x.shape)
    for steps in range(max_steps):
        g = np.asarray(gradf(x))
        m = omega1*m + (1 - omega1)*g
        v = omega2*v + (1-omega2)*np.multiply(g,g)
        hat_v = np.abs(v/(1 - omega2))
        hat_m = m/(1 - omega1)
        x = x - gamma * np.ones(shape = g.shape)/np.sqrt(hat_v + epsilon1)*hat_m
        if np.linalg.norm(g) < epsilon:
            break
    return x, v, m

def main():
    f = lambda x: x[0]**2 + x[1]**2
    gradf = lambda x: np.array([2*x[0], 2*x[1]])

    # x = steepest_descent(gradf, [1, 1])
    # x = steepest_descent_with_momentum(gradf, [1, 1])
    x, _, _ = adam_v(gradf, [1, 1])

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