import matplotlib.pyplot as plt


def plot3D(x1, x2, y, y_hat):
    """Quick function to look at the Franke function in 3D"""
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.scatter(x1, x2, y)
    ax.plot_trisurf(
        x1, x2, y_hat, linewidth=0, antialiased=False)
