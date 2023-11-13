from PIL import Image

def combine_gifs(input_files, output_file):
    images = [Image.open(file) for file in input_files]

    # Make sure all images have the same size
    sizes = set(im.size for im in images)
    if len(sizes) != 1:
        raise ValueError("All input images must have the same dimensions.")

    # Combine the images
    images[0].save(output_file, save_all=True, append_images=images[1:], loop=0)

# List of input GIF files
input_files = ["animation1.gif", "animation2.gif", "animation3.gif", "animation4.gif"]

# Output file name
output_file = "output.gif"

# Combine the GIFs
combine_gifs(input_files, output_file)