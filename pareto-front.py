import matplotlib.pyplot as plt
import numpy as np

# Generate random sample data for Objective 1 and Objective 2
all_points = np.random.normal(0, 5, (70, 2))
all_points[all_points < 0] *= -1

# Function to find inverse Pareto optimal points
def find_inverse_pareto_optimal(points):
    pareto_optimal = []
    for point in points:
        x1, y1 = point
        dominated = False
        for other_point in points:
            x2, y2 = other_point
            if (x2 >= x1 and y2 > y1) or (x2 > x1 and y2 >= y1):
                dominated = True
                break
        if not dominated:
            pareto_optimal.append(point)
    return np.array(pareto_optimal)

# Find inverse Pareto optimal points
pareto_optimal_points = find_inverse_pareto_optimal(all_points)

# Exclude Pareto optimal points from all points
non_pareto_points = np.array([point for point in all_points if not any(np.all(point == x) for x in pareto_optimal_points)])

# Separate the coordinates for plotting
non_pareto_x, non_pareto_y = non_pareto_points[:, 0], non_pareto_points[:, 1]
pareto_x, pareto_y = pareto_optimal_points[:, 0], pareto_optimal_points[:, 1]

# Create the scatter plot
fig, ax = plt.subplots(figsize=(10, 8), dpi=200)  # Set the figure size

# Plot the points
ax.scatter(non_pareto_x, non_pareto_y, c='black',marker='.', linewidths=2, s=120, label='Non-Pareto Points')
ax.scatter(pareto_x, pareto_y, c='red', marker='x', linewidths=2, s=140, label='Pareto Optimal Points')

# Remove axis labels and borders
ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
ax.spines[['top', 'right']].set_visible(False)

# Extend the axis limits
ax.set_xlim(0, 16)
ax.set_ylim(0, 16)

# Add arrows
ax.plot((1), (0), ls="", marker=">", ms=8, color="k",
        transform=ax.get_yaxis_transform(), clip_on=False)
ax.plot((0), (1), ls="", marker="^", ms=8, color="k",
        transform=ax.get_xaxis_transform(), clip_on=False)

ax.text(8, -0.7, r"$ X $", fontsize=20)
ax.text(-0.5, 8, r"$ Y $", fontsize=20)

# Show the plot
plt.show()
plt.tight_layout()
