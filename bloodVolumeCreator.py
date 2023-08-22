import numpy as np
import random
import matplotlib.pyplot as plt
from multiprocessing import Pool

def generate_spheres(grid_size, num_spheres, min_radius, max_radius, voxel_resolution):
    # Define parameters
    sphere_radii = np.random.uniform(min_radius, max_radius, num_spheres)
    
    # Initialize the 3D grid
    space = np.zeros((grid_size, grid_size, grid_size), dtype=int)
    
    # Place spheres randomly within the grid
    for radius in sphere_radii:
        
        is_overlap = True
        while is_overlap:
            sphere_center = (
                random.randint(0 + max_radius, grid_size - max_radius),
                random.randint(0 + max_radius, grid_size - max_radius),
                random.randint(0 + max_radius, grid_size - max_radius)
            )
            
            is_overlap = False
            int_radius = int(radius)
            for x in range(max(0, sphere_center[0] - int_radius), min(grid_size, sphere_center[0] + int_radius + 1)):
                for y in range(max(0, sphere_center[1] - int_radius), min(grid_size, sphere_center[1] + int_radius + 1)):
                    for z in range(max(0, sphere_center[2] - int_radius), min(grid_size, sphere_center[2] + int_radius + 1)):
                        distance = np.sqrt((x - sphere_center[0])**2 + (y - sphere_center[1])**2 + (z - sphere_center[2])**2)
                        if distance <= radius:
                            if space[x, y, z] == 1:
                                is_overlap = True
                                break
                    if is_overlap:
                        break
                if is_overlap:
                    break
            
            # Update the space with the new sphere
            int_voxel_resolution = int(voxel_resolution * radius)
            for x in range(max(0, sphere_center[0] - int_radius), min(grid_size, sphere_center[0] + int_radius + 1)):
                for y in range(max(0, sphere_center[1] - int_radius), min(grid_size, sphere_center[1] + int_radius + 1)):
                    for z in range(max(0, sphere_center[2] - int_radius), min(grid_size, sphere_center[2] + int_radius + 1)):
                        distance = np.sqrt((x - sphere_center[0])**2 + (y - sphere_center[1])**2 + (z - sphere_center[2])**2)
                        if distance <= radius:
                            space[x, y, z] = 1
    
    # Generate 2D slices along the x-axis
    x_slices = [space[x, :, :] for x in range(grid_size)]
    
    return space, x_slices

def generate_toroids(grid_size, num_toroids, major_radius_range, minor_radius_range, voxel_resolution):
    # Define parameters
    toroid_major_radii = np.random.uniform(major_radius_range[0], major_radius_range[1], num_toroids)
    toroid_minor_radii = np.random.uniform(minor_radius_range[0], minor_radius_range[1], num_toroids)

    # Initialize the 3D grid
    space = np.zeros((grid_size, grid_size, grid_size), dtype=int)

    for i in range(num_toroids):
        print(i)
        is_overlap = True
        while is_overlap:
            toroid_center = np.array([
                random.randint(int(toroid_major_radii[i] / voxel_resolution), grid_size - int(toroid_major_radii[i] / voxel_resolution)),
                random.randint(int(toroid_major_radii[i] / voxel_resolution), grid_size - int(toroid_major_radii[i] / voxel_resolution)),
                random.randint(int(toroid_major_radii[i] / voxel_resolution), grid_size - int(toroid_major_radii[i] / voxel_resolution))
            ])

            x_coords, y_coords, z_coords = np.ogrid[0:grid_size, 0:grid_size, 0:grid_size]
            distance_to_major_axis = np.sqrt(((x_coords * voxel_resolution) - toroid_center[0])**2 + ((y_coords * voxel_resolution) - toroid_center[1])**2)
            distance = np.sqrt((distance_to_major_axis - toroid_major_radii[i])**2 + ((z_coords * voxel_resolution) - toroid_center[2])**2)
            toroid_voxels = distance <= toroid_minor_radii[i]
            
            # Check for overlap using any() on the existing space
            if not np.any(space[toroid_voxels]):
                space[toroid_voxels] = 1
                is_overlap = False

    # Generate 2D slices along the x-axis
    x_slices = [space[x, :, :] for x in range(grid_size)]

    return space, x_slices


def rotate_toroid_voxels(voxels, center, phi, theta, grid_size, voxel_resolution):
    x_coords, y_coords, z_coords = np.where(voxels)
    x_coords, y_coords, z_coords = x_coords * voxel_resolution, y_coords * voxel_resolution, z_coords * voxel_resolution

    # Translate to toroid center
    x_coords -= center[0] * voxel_resolution
    y_coords -= center[1] * voxel_resolution
    z_coords -= center[2] * voxel_resolution

    # Apply rotations using trigonometric functions
    x_rotated = x_coords * np.cos(theta) * np.cos(phi) - y_coords * np.sin(phi) + z_coords * np.sin(theta) * np.cos(phi)
    y_rotated = x_coords * np.cos(theta) * np.sin(phi) + y_coords * np.cos(phi) + z_coords * np.sin(theta) * np.sin(phi)
    z_rotated = -x_coords * np.sin(theta) + z_coords * np.cos(theta)

    # Translate back to original position
    x_rotated += center[0] * voxel_resolution
    y_rotated += center[1] * voxel_resolution
    z_rotated += center[2] * voxel_resolution

    # Round and clip rotated coordinates
    x_rotated = np.clip(np.round(x_rotated / voxel_resolution), 0, grid_size - 1).astype(int)
    y_rotated = np.clip(np.round(y_rotated / voxel_resolution), 0, grid_size - 1).astype(int)
    z_rotated = np.clip(np.round(z_rotated / voxel_resolution), 0, grid_size - 1).astype(int)

    # Create a mask for rotated voxels
    rotated_voxels = np.zeros_like(voxels)
    rotated_voxels[x_rotated, y_rotated, z_rotated] = True

    return rotated_voxels


def generate_toroids_with_random_rotations(grid_size, num_toroids, major_radius_range, minor_radius_range, voxel_resolution):
    # Define parameters
    
    toroid_major_radii = np.random.uniform(major_radius_range[0], major_radius_range[1], num_toroids)
    toroid_minor_radii = np.random.uniform(minor_radius_range[0], minor_radius_range[1], num_toroids)

    # Initialize the 3D grid
    space = np.zeros((grid_size, grid_size, grid_size), dtype=int)

    for i in range(num_toroids):
        print(i)
        is_overlap = True
        while is_overlap:
            toroid_center = np.array([
                random.randint(int(toroid_major_radii[i] / voxel_resolution), grid_size - int(toroid_major_radii[i] / voxel_resolution)),
                random.randint(int(toroid_major_radii[i] / voxel_resolution), grid_size - int(toroid_major_radii[i] / voxel_resolution)),
                random.randint(int(toroid_major_radii[i] / voxel_resolution), grid_size - int(toroid_major_radii[i] / voxel_resolution))
            ])

            x_coords, y_coords, z_coords = np.ogrid[0:grid_size, 0:grid_size, 0:grid_size]
            distance_to_major_axis = np.sqrt(((x_coords * voxel_resolution) - toroid_center[0])**2 + ((y_coords * voxel_resolution) - toroid_center[1])**2)
            distance = np.sqrt((distance_to_major_axis - toroid_major_radii[i])**2 + ((z_coords * voxel_resolution) - toroid_center[2])**2)
            toroid_voxels = distance <= toroid_minor_radii[i]
            
            # Check for overlap using any() on the existing space
            if not np.any(space[toroid_voxels]):
                space[toroid_voxels] = 1
                is_overlap = False

                # Apply random rotations to toroid voxels
                phi = np.random.uniform(0, 2*np.pi)
                theta = np.random.uniform(0, 2*np.pi)
                toroid_voxels_rotated = rotate_toroid_voxels(toroid_voxels, toroid_center, phi, theta, grid_size, voxel_resolution)

                space[toroid_voxels_rotated] = 1

    # Generate 2D slices along the x-axis
    x_slices = [space[x, :, :] for x in range(grid_size)]

    return space, x_slices

