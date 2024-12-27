import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
matplotlib.use('Agg')

filename = '../input/grawimetria.dat'
G = 6.674e-11
delta_rho = 1.2 - 2600
g0 = 9.81


def model_grav(x, R, h):
    m = (4 / 3) * np.pi * R ** 3 * delta_rho
    return (G * m * h) / (x**2 + h**2)**(3/2) * 1e5


try:
    data = pd.read_csv(filename, encoding='UTF-16', delimiter='\t', header=None)
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    popt, pcov = curve_fit(model_grav, x, y, p0=[1, 1])
    print(popt)
    R_opt, h_opt, = popt
    y_fit = model_grav(x, *popt)
    max_index = np.argmax(np.abs(y_fit))
    x_max = x[max_index]
    plt.plot(x, model_grav(x, R_opt, h_opt), color='red')
    plt.scatter(x, y_fit, color='black')
    plt.savefig('../output/gravity.png')

    plt.figure(figsize=(8, 8))
    plt.axhline(y=0, color='black')
    ground = mpatches.Rectangle((-6, 0), 12, -8, color='black', alpha=0.3)
    plt.gca().add_patch(ground)
    plt.xlim([-6, 6])
    plt.ylim([-8, 6])
    anomaly = mpatches.Circle((x_max, -1 * h_opt), R_opt, linewidth=3, edgecolor='brown', facecolor='brown', alpha=0.3)
    plt.gca().add_patch(anomaly)
    plt.savefig('../output/anomaly.png')

except FileNotFoundError:
    print(f'Could not read file')



