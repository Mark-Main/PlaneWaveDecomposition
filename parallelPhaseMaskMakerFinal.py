import numpy as np
import os
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt
import bloodGenerator
import sphereParticleGenerator

# Parameters
waveRes = 500
sliceThickness = 20
volume_dimensions = (waveRes, waveRes, sliceThickness)
discocyte_radius = 16
num_normal_discocytes = 35
num_bumpy_discocytes = 105
num_crescents = 0
bump_radius = 8
number_of_bumps = 15
cellSize = 8e-6
shape_type = 'sphere'
sphere_radius = 6  # Global constant, sphere radius that never changes

def generate_and_save_phase_screen(slice_number, base_directory, num_spheres, sphere_radius, num_ellipsoids, shape_type='sphere'):
    if shape_type == 'disc':
        simulated_volume, _ = bloodGenerator.simulate_discocytes(
            volume_dimensions, discocyte_radius, num_normal_discocytes, num_bumpy_discocytes, num_crescents, bump_radius, number_of_bumps)
    elif shape_type == 'sphere':
        ellipsoid_dimensions = (np.random.randint(5, 20), np.random.randint(5, 20), np.random.randint(1, 5))
        simulated_volume = sphereParticleGenerator.simulate_objects(volume_dimensions, num_spheres, sphere_radius, num_ellipsoids, ellipsoid_dimensions)

    n_white = 1.5
    n_non_white = 1.3
    wavelength = 850e-9
    voxel_size = 0.25e-6

    refractive_indices = np.where(simulated_volume == 1, n_white, n_non_white)
    OPL = refractive_indices * voxel_size
    phase_shift = (2 * np.pi / wavelength) * OPL
    phase_screen = np.sum(phase_shift, axis=2)

    output_directory = os.path.join(base_directory, f'PhaseScreens{shape_type.capitalize()}')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    file_path = os.path.join(output_directory, f'phase_screen_{slice_number}.npy')
    np.save(file_path, phase_screen)

    #  # Plot the phase screen
    # plt.figure(figsize=(8, 6))
    # plt.imshow(phase_screen, cmap='gray')
    # plt.colorbar(label='Phase Shift')
    # plt.title(f'Phase Screen {slice_number} for {shape_type}')
    # plt.show()
    
    return f'Phase screen {slice_number} saved for {shape_type}'

def worker(config):
    slice_number, base_directory, num_spheres, num_ellipsoids, shape_type = config
    return generate_and_save_phase_screen(slice_number, base_directory, num_spheres, sphere_radius, num_ellipsoids, shape_type)

if __name__ == '__main__':
    configurations = [
        (r'/Volumes/DeBroglie/Sphere100/', 1000, 0),
        (r'/Volumes/DeBroglie/Sphere90/', 900, 100),
        (r'/Volumes/DeBroglie/Sphere98/', 980, 20),
        (r'/Volumes/DeBroglie/Sphere99/', 990, 10),
        (r'/Volumes/DeBroglie/Sphere99_9/', 999, 1),
    ]
    num_processes = min(10, multiprocessing.cpu_count()-1)
    run_count = 500

    for base_directory, num_spheres, num_ellipsoids in configurations:
        pool = multiprocessing.Pool(processes=num_processes)
        args = [(i, base_directory, num_spheres, num_ellipsoids, shape_type) for i in range(run_count)]
        for _ in tqdm(pool.imap_unordered(worker, args), total=run_count, desc=f"Generating phase screens in {base_directory}", ascii=False, ncols=75):
            pass
        pool.close()
        pool.join()
