import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the cube vertices
vertices = [
    (0, 0, 0), (5, 0, 0), 
    (5, 5, 0), (0, 5, 0), 
    (0, 0, 5), (5, 0, 5), 
    (5, 5, 5), (0, 5, 5)
]

# Define the edges by connecting vertex indices
edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each edge
for i, j in edges:
    x, y, z = zip(vertices[i], vertices[j])
    ax.plot(x, y, z)

# Label the front bottom edge (from (0,0,0) to (5,0,0))
mid = ((0 + 5) / 2, 0, 0)
ax.text(mid[0], mid[1], mid[2], '5 cm')

# Ensure all axes have equal scale
ax.set_box_aspect((1, 1, 1))

# Turn off axis lines and ticks for a cleaner look
ax.set_axis_off()

plt.show()
