import numpy as np
import matplotlib.pyplot as plt
from pymoo.factory import get_performance_indicator
from scipy.interpolate import make_interp_spline

def plot_pareto_front_with_hypervolume(non_dominated, reference_point, dominated, pareto):
    non_dominated = non_dominated[np.argsort(non_dominated[:, 0])]

    fig, ax = plt.subplots(figsize=(7, 7), dpi=200)  # Set the figure size
    
    # Plot the hypervolume rectangles
    for point in non_dominated:
        plt.fill_between([point[0], reference_point[0]], [point[1], point[1]], [reference_point[1], reference_point[1]],
                         color='skyblue', alpha=1, step='post')

    # Plot the reference point
    plt.scatter(reference_point[0], reference_point[1], color='red', label='Reference Point', zorder=5, marker="x")

    # Set the plot limits
    plt.xlim(0, max(reference_point[0], np.max(pareto[:, 0])) + 1)
    plt.ylim(0, max(reference_point[1], np.max(pareto[:, 1])) + 1)

    # Add labels and legend
    plt.title('')
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')

    spline = make_interp_spline(pareto[:, 0], pareto[:, 1], k=3)
    xnew = np.linspace(pareto[:, 0].min(), pareto[:, 0].max(), 300) 
    ynew = spline(xnew)

    spline = make_interp_spline(non_dominated[:, 0], non_dominated[:, 1], k=2)  
    xnew_2 = np.linspace(non_dominated[:, 0].min(), non_dominated[:, 0].max(), 30)
    ynew_2 = spline(xnew_2)

    # Plot the Pareto front line
    plt.plot(xnew, ynew, label='Actual Pareto Front', color='green')
    plt.plot(xnew_2, ynew_2, label='Estimated Pareto front', color='blue')
    plt.scatter(dominated[:, 0], dominated[:, 1], label='Dominated policies', color='black', marker='x')
    plt.scatter(non_dominated[:, 0], non_dominated[:, 1], label='Non-dominated policies', color='blue', marker='o')

    # Add arrows
    ax.plot((1), (0), ls="", marker=">", ms=8, color="k",
            transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot((0), (1), ls="", marker="^", ms=8, color="k",
            transform=ax.get_xaxis_transform(), clip_on=False)

    ax.spines[['top', 'right']].set_visible(False)
    plt.legend()
    plt.show()

pareto = np.array([[1,6.6], [2,6.4], [3, 6], [4.5, 5], [5, 4.5], [6, 3],[6.4,1.9], [6.6, 1]])
non_dominated = np.array([[1.5, 5], [3.2, 4.2], [4.2, 3.2], [5, 1.7]])
dominated =  np.array([
    [1.0, 4.5],
    [1.5, 3.8],
    [2.0, 3.0],
    [1.6, 2.2],
    [4.0, 1.5],
    [1.8, 1.2],
    [3.5, 3.1],
    [2.0, 1.8],
    [2.4, 3.2],
    [3.9, 1.7],
    [1.9, 3.3],
    [4.5, 1.5],
    [3.0, 1.2]])
reference_point = [1, 1]

plot_pareto_front_with_hypervolume(non_dominated, reference_point, dominated, pareto)
