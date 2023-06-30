import numpy as np

def disStep(d, res, s, λ):
    # Create an empty complex array of size (res, res)
    result = np.zeros((res, res), dtype=complex)

    # Iterate over the indices x and y
    for x in range(res):
        for y in range(res):
            # Term 1: First component of the distance factor
            # It represents the square of (2 * π / λ)
            term1 = (6.28318 / λ) ** 2

            # Term 2: Second component of the distance factor
            # It calculates the horizontal distance from the center to the current point
            # by wrapping the indices around within the range of res
            # The distance is scaled by s and squared
            term2 = (2 * np.pi * ((x + res / 2) % res - res / 2) / s) ** 2

            # Term 3: Third component of the distance factor
            # It calculates the vertical distance from the center to the current point
            # by wrapping the indices around within the range of res
            # The distance is scaled by s and squared
            term3 = (2 * np.pi * ((y + res / 2) % res - res / 2) / s) ** 2

            # Calculate the distance factor by subtracting term2 and term3 from term1 and taking the square root
            distance_factor = np.sqrt(term1 - term2 - term3)

            # Compute the complex value by multiplying with d and exponentiating with (0 + 1j)
            result[x, y] = d * np.exp(distance_factor * (0 + 1j))

    return result
