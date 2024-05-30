import collada


def calculate_dimensions(vertices):
    min_x = min(vertices, key=lambda v: v[0])[0]
    max_x = max(vertices, key=lambda v: v[0])[0]
    min_y = min(vertices, key=lambda v: v[1])[1]
    max_y = max(vertices, key=lambda v: v[1])[1]
    min_z = min(vertices, key=lambda v: v[2])[2]
    max_z = max(vertices, key=lambda v: v[2])[2]
    return (max_x - min_x, max_y - min_y, max_z - min_z)

def get_model_dimensions(dae_file):
    # Load the Collada file
    try:
        collada_mesh = collada.Collada(dae_file)
    except collada.ColladaError as e:
        print("Error loading Collada file:", e)
        return None

    # Get the scene from the Collada file
    scene = collada_mesh.scene

    # Iterate over the geometries in the scene
    for geometry in scene.objects('geometry'):
        for primitive in geometry.primitives():
            if isinstance(primitive, collada.triangleset.BoundTriangleSet):
                # Extract vertices from the geometry
                vertices = primitive.vertex
                # Convert the vertices to a list of tuples
                vertices = [(float(v[0]), float(v[1]), float(v[2])) for v in vertices]
                # Calculate dimensions
                dimensions = calculate_dimensions(vertices)
                return dimensions

# Example usage
dae_file = "/home/kedhar/Desktop/large_orange_cone.dae"
dimensions_dae = get_model_dimensions(dae_file)
print(dimensions_dae)
# Scaling factors applied in Gazebo
gazebo_scale = (0.254, 0.254, 0.254)

# Scale the dimensions according to Gazebo
dimensions_gazebo = tuple(dim * scale for dim, scale in zip(dimensions_dae, gazebo_scale))

print("Scaled dimensions for Gazebo:", dimensions_gazebo)

model_position = (-103.872,-20.2191,0.714453)

print('model pose',model_position)
# Calculate half of the scaled dimensions
half_dimensions = tuple(dim / 2 for dim in dimensions_gazebo)

# Calculate the minimum and maximum coordinates of the bounding box
min_coords = (model_position[0] - half_dimensions[0], model_position[1] - half_dimensions[1], model_position[2] - half_dimensions[2])
max_coords = (model_position[0] + half_dimensions[0], model_position[1] + half_dimensions[1], model_position[2] + half_dimensions[2])

print("Bounding box minimum coordinates:", min_coords)
print("Bounding box maximum coordinates:", max_coords)

# Compute the coordinates of the eight corners
corners = [
    (min_coords[0], min_coords[1], min_coords[2]),
    (max_coords[0], min_coords[1], min_coords[2]),
    (max_coords[0], max_coords[1], min_coords[2]),
    (min_coords[0], max_coords[1], min_coords[2]),
    (min_coords[0], min_coords[1], max_coords[2]),
    (max_coords[0], min_coords[1], max_coords[2]),
    (max_coords[0], max_coords[1], max_coords[2]),
    (min_coords[0], max_coords[1], max_coords[2])
]

# Print the coordinates of each corner
for i, corner in enumerate(corners):
    print(f"Corner {i + 1}: {corner}")