import numpy as np
import csv
import matplotlib.pyplot as plt

class LoopySim:
    def __init__(self, lobes=4, amplitude=0.7, N=36, scale=10):
        self.lobes = lobes          # number of lobes in the morphology
        self.amplitude = amplitude  # how pronounced the lobes are 
        self.N = N                  # number of points in the morphology i.e. number of light sensors
        self.scale = scale          # scales the size of loopy
        

    def generate_theta(self):
        ''' generates the motor angle values for the Loopy robot'''

        # Ensure N is even
        if self.N % 2 != 0:
            raise ValueError("N must be an even number.")
        
        constant = 2 * np.pi / self.N
        x = np.linspace(0, 2 * np.pi * self.lobes, self.N)
        theta = constant + self.amplitude * np.sin(x)
        return theta

    def generate_loopy_coords(self):
        '''converts the motor angles to a set of xy points of the light sensors'''
        
        theta = self.generate_theta()
        
        x_coords, y_coords = [], []
        x0, y0, t0 = 0, 0, 0

        # Calculate coordinates
        for angle in theta:
            t1 = angle + t0
            x1 = np.cos(t1) + x0
            y1 = np.sin(t1) + y0
            x_coords.append(x1)
            y_coords.append(y1)   
            t0, x0, y0 = t1, x1, y1

        # scale the coordinates
        x_coords = np.array(x_coords) * self.scale
        y_coords = np.array(y_coords) * self.scale

        return x_coords, y_coords

    def prey_image_creator(self, outer_radius=5, thickness=2, pixels_per_unit=10):
        ''' generate a prey image'''
        
        # calculate the pixel values for the prey image
        outer_radius_px = int(outer_radius * pixels_per_unit)
        thickness_px = int(thickness * pixels_per_unit)
        inner_radius_px = outer_radius_px - thickness_px
        image_size = 2 * outer_radius_px
        center = outer_radius_px

        # initalize blank image
        image = np.zeros((image_size, image_size), dtype=int)

        # draw the circle for the prey image
        for y in range(image_size):
            for x in range(image_size):
                distance = np.sqrt((x - center) ** 2 + (y - center) ** 2)
                if inner_radius_px <= distance <= outer_radius_px:
                    image[y, x] = 1

        return image

    def predator_image_creator(self, length=10, width=2, pixels_per_unit=10):
        '''create the predator image'''

        # calculate the pixel values for the predator image
        length_px = int(length * pixels_per_unit)
        width_px = int(width * pixels_per_unit)
        image_size = length_px
        center = length_px // 2
        half_width_px = width_px // 2

        # initalize blank image
        image = np.zeros((image_size, image_size), dtype=int)
        
        # draw the + with two rectangles for the predator image
        image[center - half_width_px : center + half_width_px + 1, :] = 1
        image[:, center - half_width_px : center + half_width_px + 1] = 1

        return image

    def nothing_image_creator(self, size=10, pixels_per_unit=10):
        """ Creates an all-black image """
        
        image_size = int(size * pixels_per_unit)
        image = np.zeros((image_size, image_size), dtype=int)
        return image
    
    def noise_image_creator(self, size=10, pixels_per_unit=10, probability=0.1):
        """ Creates an image with random uniform noise"""
        
        if not (0 <= probability <= 1):
            raise ValueError("Probability must be between 0 and 1.")
        
        image_size = int(size * pixels_per_unit)
        image = np.random.choice([0, 1], size=(image_size, image_size), p=[1 - probability, probability])
        return image

    def overlay_loopy_on_image(self, background_image, x_coords, y_coords, x_pos=0, y_pos=0, rotation=0, plot=True):
        ''' put loopy on the image at the given centroid pose and read the light sensor values, also return the transformed coordinates'''

        # Calculate starting centroid of Loopy coordinates
        centroid_x = np.mean(x_coords)
        centroid_y = np.mean(y_coords)

        # Transform coordinates (rotation and translation)
        transformed_x_coords = []
        transformed_y_coords = []
        rad = np.radians(rotation)
        for x, y in zip(x_coords, y_coords):
            # rotate
            rotated_x = (x - centroid_x) * np.cos(rad) - (y - centroid_y) * np.sin(rad)
            rotated_y = (x - centroid_x) * np.sin(rad) + (y - centroid_y) * np.cos(rad)
            
            # translate
            transformed_x = rotated_x + x_pos  
            transformed_y = rotated_y + y_pos  
            
            transformed_x_coords.append(transformed_x)
            transformed_y_coords.append(transformed_y)

        # transformed centroid
        transformed_centroid_x =np.mean(transformed_x_coords)
        transformed_centroid_y = np.mean(transformed_y_coords)
        
        

        # Determine light intensities at Loopy points
        light_intensity = []
        node_colors = []
        for tx, ty in zip(transformed_x_coords, transformed_y_coords):
            # convert to pixel coordinates
            pixel_x, pixel_y = int(tx), int(ty)

            # if point is whithin the image get the intensity value
            if 0 <= pixel_x < background_image.shape[1] and 0 <= pixel_y < background_image.shape[0]:
                intensity = background_image[pixel_y, pixel_x]
            else:
                intensity = 0  # outside points are considered dark

            light_intensity.append(intensity)
            node_colors.append("red" if intensity == 1 else "blue") # record colors for plotting later

        # Plot the Loopy path over the background image
        if plot:
            fig, ax = plt.subplots()

            # plot background image: prey, predator, noise
            ax.imshow(background_image, cmap="gray", origin="upper", vmin=0, vmax=1)

            # plot Loopy points with color based on light intensity: red = bright (1), blue = dark (0)
            ax.scatter(transformed_x_coords, transformed_y_coords, c=node_colors, edgecolors="black", s=50)
            ax.plot(transformed_x_coords, transformed_y_coords, color="green", linewidth=1)

            # plot the centroid
            ax.scatter([transformed_centroid_x], [transformed_centroid_y], c="red", marker="x", s=100, label="Centroid")

            # Set plot limits
            ax.set_xlim(0, background_image.shape[1])
            ax.set_ylim(background_image.shape[0], 0)  # flip y-axis for image orientation
            ax.legend()
            plt.show()

        # Return the light intensity vector and transformed loopy coordinates
        light_intensity = np.array(light_intensity)
        return light_intensity, transformed_x_coords, transformed_y_coords
    
    def loopy_dataset_generator(self, dx, dy, dtheta, filename='loopy_dataset.csv'):
        ''' generate light dataset for prey, predator and noise images across all poses'''
        
        x_coords, y_coords = self.generate_loopy_coords()

        # Open the CSV file for writing
        with open(filename, mode='w', newline='') as file:
           
            writer = csv.writer(file)
            
            # Write the header
            header = ['x_shift', 'y_shift', 'rotation', 'colors', 'label']
            writer.writerow(header)

            # Create a prey, predator, and noise images
            prey_image = self.prey_image_creator()
            predator_image = self.predator_image_creator()
            noise_image = self.noise_image_creator()

            images = {
                "prey": prey_image,
                "predator": predator_image,
                "noise": noise_image,
            }

            # for each image, x position, y position and rotation angle generate the light sensor values

            for image_type, image in images.items():
                # Step through the space and rotation angle
                for x_shift in np.arange(0, image.shape[1], dx):
                    for y_shift in np.arange(0, image.shape[0], dy):
                        for angle in np.arange(0, 360, dtheta):
                            
                            if image_type == "noise":   # Noise image is generated randomly each time
                                image = self.noise_image_creator()

                            colors,tx,ty = self.overlay_loopy_on_image(image, x_coords, y_coords, x_pos=x_shift, y_pos=y_shift, rotation=angle, plot=False)
                            label = image_type

                            # save to csv file
                            writer.writerow([x_shift, y_shift, angle, list(colors), label])

        print(f"Dataset saved to {filename}")

    
    def visualize_label_from_csv(self, filename, label, num_samples=100, plot_height=100, save_image=False):
        ''' visualize a large chunk of the data set for a given label'''
        
        # load csv file and extract the light sensor values for the given label only
        data = []
        with open(filename, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['label'] == label:
                    colors = eval(row['colors'])
                    data.append(colors)
                    if len(data) >= num_samples:
                        break
        
        if not data:
            print(f"No data found for label '{label}'.")
            return
        
        # convert the data to an array and transpose it so it is wider than tall
        image_data = np.array(data).T
        
        # display the image
        plot_width = num_samples / 2
        plt.figure(figsize=(plot_width, plot_height), dpi=5)
        plt.imshow(image_data, cmap='gray', vmin=0, vmax=1, aspect='auto')
        plt.axis('off')
        plt.title(f"Visualization for label '{label}' (first {num_samples} samples, transposed)")

        # save image to file
        if save_image:
            plt.savefig(f"{label}_visualization.png", bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        return data