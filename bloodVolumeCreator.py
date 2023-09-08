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

def generate_torus(R, r, num_points):
    phi = np.linspace(0, 2*np.pi, num_points)
    theta = np.linspace(0, 2*np.pi, num_points)
    phi, theta = np.meshgrid(phi, theta)
    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    return x, y, z

def generate_voxelized_toroids(grid_size, num_toroids, R, r, num_points=100):
    toroid_positions = []
    voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=int)

    for _ in range(num_toroids):
        print("Generating toroid:", _)
        toroid_collides = True
        phi = np.random.uniform(0, 2*np.pi)
        theta = np.random.uniform(0, 2*np.pi)

        for i in range(0,R):
            print(i)
            while toroid_collides:
                x, y, z = generate_torus(R, r-i, num_points)

                x_rotated = x * np.cos(theta) * np.cos(phi) - y * np.sin(phi) * np.cos(theta) + z * np.sin(theta)
                y_rotated = x * np.sin(phi) + y * np.cos(phi)
                z_rotated = -x * np.sin(theta) * np.cos(phi) + y * np.sin(theta) * np.sin(phi) + z * np.cos(theta)

                x_scaled = x_rotated / 5
                y_scaled = y_rotated / 5
                z_scaled = z_rotated / 5

                x_offset = np.random.uniform(-(grid_size-R), (grid_size-R))
                y_offset = np.random.uniform(-(grid_size-R), (grid_size-R))
                z_offset = np.random.uniform(-(grid_size-R), (grid_size-R))

                x_final = x_scaled + x_offset
                y_final = y_scaled + y_offset
                z_final = z_scaled + z_offset

                toroid_collides = any(
                    np.linalg.norm(np.array([x_final, y_final, z_final]) - pos) < 2 * r
                    for pos in toroid_positions
                )

                toroid_positions.append(np.array([x_final, y_final, z_final]))

                x_indices = np.clip((x_final + grid_size / 2).astype(int), 0, grid_size - 1)
                y_indices = np.clip((y_final + grid_size / 2).astype(int), 0, grid_size - 1)
                z_indices = np.clip((z_final + grid_size / 2).astype(int), 0, grid_size - 1)
                voxel_grid[x_indices, y_indices, z_indices] = 1

    # Generate 2D slices along the x-axis
    x_slices = [voxel_grid[x, :, :] for x in range(grid_size)]        

    return voxel_grid, x_slices