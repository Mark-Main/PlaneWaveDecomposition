import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import binary_dilation, binary_erosion
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import os

# Initialize volume
def initialize_volume(dimensions):
    return np.zeros(dimensions, dtype=bool)

# Model discocyte
def model_discocyte(radius):
    dim = np.arange(-radius, radius + 1)
    X, Y, Z = np.meshgrid(dim, dim, dim)
    discocyte = (X**2 + Y**2 + (1.5 * Z)**2) <= radius**2
    return discocyte

# Add rounded bumps to the surface of the discocyte
def add_rounded_bumps_to_surface(base_model, bump_radius, number_of_bumps):
    bumpy_model = np.copy(base_model)
    surface_voxels = np.argwhere(binary_erosion(base_model) & ~binary_erosion(base_model, structure=np.ones((3,3,3))))
    bump_bases = surface_voxels[np.random.choice(surface_voxels.shape[0], number_of_bumps, replace=False)]

    for base in bump_bases:
        for x in range(-bump_radius, bump_radius + 1):
            for y in range(-bump_radius, bump_radius + 1):
                for z in range(-bump_radius, bump_radius + 1):
                    if x**2 + y**2 + z**2 <= bump_radius**2:
                        bump_position = base + np.array([x, y, z])
                        bump_position = np.clip(bump_position, 0, np.array(base_model.shape) - 1).astype(int)
                        bumpy_model[tuple(bump_position)] = True

    return bumpy_model

# Generate random orientation
def random_orientation():
    rotation = R.random(random_state=np.random.RandomState(seed=None)).as_matrix()
    return rotation

# Rotate discocyte
def rotate_discocyte(discocyte, rotation_matrix):
    filled_indices = np.argwhere(discocyte)
    center_offset = np.array(discocyte.shape) // 2
    filled_indices_centered = filled_indices - center_offset
    rotated_indices = np.dot(filled_indices_centered, rotation_matrix).astype(int)
    rotated_indices += center_offset
    rotated_discocyte = np.zeros_like(discocyte)
    for ind in rotated_indices:
        if all(0 <= ind[i] < discocyte.shape[i] for i in range(3)):
            rotated_discocyte[tuple(ind)] = True
    return rotated_discocyte

# Crescent function
def crescent_moderate_hole(discocyte, radius):
    # The radius for the inner cylinder to be removed, making it a crescent
    inner_cylinder_radius = int(radius * 0.7)

    # Coordinates for the center of the discocyte
    center_x, center_y, center_z = np.array(discocyte.shape) // 2

    # Iterate through each voxel in the discocyte
    for x in range(discocyte.shape[0]):
        for y in range(discocyte.shape[1]):
            for z in range(discocyte.shape[2]):
                # Carve out the inner cylinder
                if (x - center_x)**2 + (y - center_y)**2 < inner_cylinder_radius**2:
                    discocyte[x, y, z] = False
                # Keep only the half where z <= center_z
                elif z > center_z:
                    discocyte[x, y, z] = False

    return discocyte


# Place discocyte in volume
def place_discocyte(volume, discocyte, position):
    radius = discocyte.shape[0] // 2
    start_positions = np.maximum(0, np.array(position) - radius)
    end_positions = np.minimum(np.array(volume.shape), np.array(position) + radius + 1)
    volume_slice = tuple(slice(start, end) for start, end in zip(start_positions, end_positions))
    discocyte_slice = tuple(slice(max(0, radius - position[i]), min(discocyte.shape[i], radius - position[i] + volume.shape[i])) for i in range(3))
    volume[volume_slice] |= discocyte[discocyte_slice]
   

# Check for collision
def check_collision(volume, discocyte, position):
    # Ensure discocyte is a 3D array
    if discocyte.ndim != 3:
        raise ValueError("Discocyte shape is not 3D. Current shape: {}".format(discocyte.shape))

    radius = discocyte.shape[0] // 2
    start_positions = np.maximum(0, np.array(position) - radius)
    end_positions = np.minimum(np.array(volume.shape), np.array(position) + radius + 1)

    discocyte_slice = tuple(slice(max(0, radius - position[i]), min(discocyte.shape[i], radius - position[i] + volume.shape[i])) for i in range(3))

    return np.any(volume[tuple(slice(start, end) for start, end in zip(start_positions, end_positions))] & discocyte[discocyte_slice])


# Morphological closing
def morphological_closing(volume, structure=None):
    dilated_volume = binary_dilation(volume, structure=structure)
    closed_volume = binary_erosion(dilated_volume, structure=structure)
    return closed_volume

# Simulate discocytes with morphological closing, with an option for normal and bumpy cells
def simulate_discocytes(volume_dimensions, discocyte_radius, num_normal_discocytes, num_bumpy_discocytes, num_crescents, bump_radius, number_of_bumps):
    volume = initialize_volume(volume_dimensions)
    structure = np.ones((3, 3, 3))

    # Simulate normal discocytes
    for _ in range(num_normal_discocytes):
        collision = True
        while collision:
            position = (np.random.randint(0, volume_dimensions[0]),
                        np.random.randint(0, volume_dimensions[1]),
                        np.random.randint(0, volume_dimensions[2]))
            rotation_matrix = random_orientation()
            discocyte = model_discocyte(discocyte_radius)
            rotated_discocyte = rotate_discocyte(discocyte, rotation_matrix)
            collision = check_collision(volume, rotated_discocyte, position)
            if not collision:
                place_discocyte(volume, rotated_discocyte, position)
                volume = morphological_closing(volume, structure)

    # Simulate bumpy discocytes
    for _ in range(num_bumpy_discocytes):
        
        collision = True
        while collision:
            position = (np.random.randint(0, volume_dimensions[0]),
                        np.random.randint(0, volume_dimensions[1]),
                        np.random.randint(0, volume_dimensions[2]))
            rotation_matrix = random_orientation()
            discocyte = model_discocyte(discocyte_radius)
            rotated_discocyte = rotate_discocyte(discocyte, rotation_matrix)
            bumpy_discocyte = add_rounded_bumps_to_surface(rotated_discocyte, bump_radius, number_of_bumps)
            collision = check_collision(volume, bumpy_discocyte, position)
            if not collision:
                place_discocyte(volume, bumpy_discocyte, position)
                volume = morphological_closing(volume, structure)

    # In the simulate_discocytes function, where crescents are being generated
    for _ in range(num_crescents):
       
        collision = True
        while collision:
            position = (np.random.randint(0, volume_dimensions[0]),
                        np.random.randint(0, volume_dimensions[1]),
                        np.random.randint(0, volume_dimensions[2]))
            
            # Generate a discocyte model and shape it into a crescent
            discocyte = model_discocyte(discocyte_radius)
            crescent_discocyte = crescent_moderate_hole(discocyte, discocyte_radius)

            # Generate a random orientation and apply it to the crescent_discocyte
            rotation_matrix = random_orientation()
            crescent_discocyte = rotate_discocyte(crescent_discocyte, rotation_matrix)

            # Check if crescent_discocyte is 3D
            if crescent_discocyte.ndim != 3:
                raise ValueError("Crescent discocyte shape is not 3D. Current shape: {}".format(crescent_discocyte.shape))

            collision = check_collision(volume, crescent_discocyte, position)
            if not collision:
                place_discocyte(volume, crescent_discocyte, position)
                volume = morphological_closing(volume, structure)

    # Extract slices of the volume after simulation is complete
    slices = [volume[:, :, i] for i in range(volume.shape[2])]

    return volume, slices
