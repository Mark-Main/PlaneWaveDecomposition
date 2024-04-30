import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import binary_dilation, binary_erosion
import random

# Initialize volume
def initialize_volume(dimensions):
    return np.zeros(dimensions, dtype=bool)

# Model sphere
def model_sphere(radius):
    dim = np.arange(-radius, radius + 1)
    X, Y, Z = np.meshgrid(dim, dim, dim)
    sphere = (X**2 + Y**2 + Z**2) <= radius**2
    return sphere

# Model ellipsoid
def model_ellipsoid(a, b, c):
    dim_x = np.arange(-a, a + 1)
    dim_y = np.arange(-b, b + 1)
    dim_z = np.arange(-c, c + 1)
    X, Y, Z = np.meshgrid(dim_x, dim_y, dim_z)
    ellipsoid = (X**2 / a**2) + (Y**2 / b**2) + (Z**2 / c**2) <= 1
    return ellipsoid

# Generate random orientation
def random_orientation():
    rotation = R.random(random_state=np.random.RandomState(seed=None)).as_matrix()
    return rotation

# Rotate ellipsoid
def rotate_ellipsoid(ellipsoid, rotation_matrix):
    filled_indices = np.argwhere(ellipsoid)
    center_offset = np.array(ellipsoid.shape) // 2
    filled_indices_centered = filled_indices - center_offset
    rotated_indices = np.dot(filled_indices_centered, rotation_matrix).astype(int)
    rotated_indices += center_offset
    rotated_ellipsoid = np.zeros_like(ellipsoid)
    for ind in rotated_indices:
        if all(0 <= ind[i] < ellipsoid.shape[i] for i in range(3)):
            rotated_ellipsoid[tuple(ind)] = True
    return rotated_ellipsoid

# Place object in volume
def place_object(volume, obj, position):
    start_positions = np.maximum(0, np.array(position) - np.array(obj.shape) // 2)
    end_positions = np.minimum(np.array(volume.shape), np.array(position) + np.array(obj.shape) // 2 + 1)
    volume_slice = tuple(slice(start, end) for start, end in zip(start_positions, end_positions))
    obj_slice = tuple(slice(max(0, obj.shape[i] // 2 - position[i]), min(obj.shape[i], obj.shape[i] // 2 - position[i] + volume.shape[i])) for i in range(3))
    volume[volume_slice] |= obj[obj_slice]

# Check for collision
def check_collision(volume, obj, position):
    if obj.ndim != 3:
        raise ValueError("Object shape is not 3D. Current shape: {}".format(obj.shape))
    start_positions = np.maximum(0, np.array(position) - np.array(obj.shape) // 2)
    end_positions = np.minimum(np.array(volume.shape), np.array(position) + np.array(obj.shape) // 2 + 1)
    obj_slice = tuple(slice(max(0, obj.shape[i] // 2 - position[i]), min(obj.shape[i], obj.shape[i] // 2 - position[i] + volume.shape[i])) for i in range(3))
    return np.any(volume[tuple(slice(start, end) for start, end in zip(start_positions, end_positions))] & obj[obj_slice])

# Morphological closing
def morphological_closing(volume, structure=None):
    dilated_volume = binary_dilation(volume, structure=structure)
    closed_volume = binary_erosion(dilated_volume, structure=structure)
    return closed_volume

# Simulate objects (spheres and ellipsoids) in volume with collisions and morphological closing
def simulate_objects(volume_dimensions, num_spheres, sphere_radius, num_ellipsoids, ellipsoid_dimensions):
    volume = initialize_volume(volume_dimensions)
    structure = np.ones((3, 3, 3))

    # Simulate spheres
    for _ in range(num_spheres):
        collision = True
        while collision:
            position = (np.random.randint(0, volume_dimensions[0]),
                        np.random.randint(0, volume_dimensions[1]),
                        np.random.randint(0, volume_dimensions[2]))
            sphere = model_sphere(sphere_radius)
            collision = check_collision(volume, sphere, position)
            if not collision:
                place_object(volume, sphere, position)
                volume = morphological_closing(volume, structure)

    # Simulate ellipsoids
    for _ in range(num_ellipsoids):
        collision = True
        while collision:
            position = (np.random.randint(0, volume_dimensions[0]),
                        np.random.randint(0, volume_dimensions[1]),
                        np.random.randint(0, volume_dimensions[2]))
            ellipsoid = model_ellipsoid(*ellipsoid_dimensions)
            rotation_matrix = random_orientation()
            rotated_ellipsoid = rotate_ellipsoid(ellipsoid, rotation_matrix)
            collision = check_collision(volume, rotated_ellipsoid, position)
            if not collision:
                place_object(volume, rotated_ellipsoid, position)
                volume = morphological_closing(volume, structure)

    return volume
