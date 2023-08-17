import numpy as np
import random
import matplotlib.pyplot as plt

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
    
    # Place toroids randomly within the grid
    for i in range(num_toroids):
        is_overlap = True
        while is_overlap:
            toroid_center = (
                random.randint(0 + int(toroid_major_radii[i]), grid_size - int(toroid_major_radii[i])),
                random.randint(0 + int(toroid_major_radii[i]), grid_size - int(toroid_major_radii[i])),
                random.randint(0 + int(toroid_major_radii[i]), grid_size - int(toroid_major_radii[i]))
            )
            
            is_overlap = False
            for x in range(grid_size):
                for y in range(grid_size):
                    for z in range(grid_size):
                        distance_to_major_axis = np.sqrt((x - toroid_center[0])**2 + (y - toroid_center[1])**2)
                        distance = np.sqrt((distance_to_major_axis - toroid_major_radii[i])**2 + (z - toroid_center[2])**2)
                        if distance <= toroid_minor_radii[i]:
                            if space[x, y, z] == 1:
                                is_overlap = True
                                break
                    if is_overlap:
                        break
                if is_overlap:
                    break
            
            # Update the space with the new toroid
            int_voxel_resolution = int(voxel_resolution * max(toroid_major_radii[i], toroid_minor_radii[i]))
            for x in range(grid_size):
                for y in range(grid_size):
                    for z in range(grid_size):
                        distance_to_major_axis = np.sqrt((x - toroid_center[0])**2 + (y - toroid_center[1])**2)
                        distance = np.sqrt((distance_to_major_axis - toroid_major_radii[i])**2 + (z - toroid_center[2])**2)
                        if distance <= toroid_minor_radii[i]:
                            space[x, y, z] = 1
    
    # Generate 2D slices along the x-axis
    x_slices = [space[x, :, :] for x in range(grid_size)]
    
    return space, x_slices
