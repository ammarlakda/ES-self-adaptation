import numpy as np
import matplotlib.pyplot as plt


def image_fitness(image_vector, target_image):
    """
    Calculate fitness for an image vector. 
    Lower is better (we use MSE if there's a target; 
    otherwise, we use negative variance).
    """
    if target_image is None:
        # If no target, just minimize negative variance => maximize variance
        return -np.var(image_vector)
    else:
        # Compare to target using Mean Squared Error
        return np.mean((image_vector - np.ravel(target_image)) ** 2)


def initialize_population(mu, img_size, bounds, sigma_init):
    """
    Initialize population of images and mutation parameters.
    
    :param mu: population size
    :param img_size: (height, width)
    :param bounds: (min_val, max_val) for pixel intensities
    :param sigma_init: initial mutation step size
    :return: tuple (population, sigma), each with shape (mu, dim)
    """
    dim = img_size[0] * img_size[1]
    population = np.random.uniform(bounds[0], bounds[1], (mu, dim))
    sigma = np.full((mu, dim), sigma_init)
    return population, sigma


def mutate(parent, parent_sigma, dim, bounds, lr, sigma_bound):
    """
    Mutate image with self-adapting mutation rates.
    
    :param parent: 1D array of pixel intensities
    :param parent_sigma: 1D array of mutation step sizes
    :param dim: number of pixels
    :param bounds: (min_val, max_val) for pixel intensities
    :param lr: learning rate for step-size adaptation
    :param sigma_bound: lower bound on mutation step size
    :return: tuple (child, child_sigma)
    """
    # Update step size with log-normal self-adaptation
    child_sigma = parent_sigma * np.exp(lr * np.random.normal(0, 1, size=dim))
    # Ensure the step size doesn't go below sigma_bound
    child_sigma = np.clip(child_sigma, sigma_bound, None)
    # Generate child by sampling using the newly updated child_sigma
    child = parent + np.random.normal(0, child_sigma, size=dim)
    # Clip pixel intensities to valid range
    child = np.clip(child, bounds[0], bounds[1])
    return child, child_sigma


def generate_offspring(population, sigma, mu, lambd, dim, bounds, lr, sigma_bound, fitness_fn):
    """
    Generate new offspring population from current parents.
    
    :param population: shape (mu, dim)
    :param sigma: shape (mu, dim)
    :param mu: population size
    :param lambd: number of offspring
    :param dim: number of pixels in each individual
    :param bounds: (min_val, max_val) for pixel intensities
    :param lr: learning rate for step-size adaptation
    :param sigma_bound: lower bound on mutation step size
    :param fitness_fn: callable to evaluate individual fitness
    :return: list of tuples (child, child_sigma, fitness)
    """
    offspring = []
    for _ in range(lambd):
        # Randomly pick a parent
        parent_idx = np.random.randint(mu)
        parent = population[parent_idx]
        parent_sigma = sigma[parent_idx]

        # Generate one child
        child, child_sigma = mutate(parent, parent_sigma, dim, bounds, lr, sigma_bound)
        fitness = fitness_fn(child)
        offspring.append((child, child_sigma, fitness))
    return offspring


def select_survivors(offspring, mu):
    """
    Select top mu individuals from offspring based on fitness.
    
    :param offspring: list of tuples (child, child_sigma, fitness)
    :param mu: desired number of survivors
    :return: (new_population, new_sigma, best_solution, best_fitness)
    """
    # Sort by fitness (ascending => best = lowest fitness)
    offspring.sort(key=lambda x: x[2])
    # Take the best mu
    best_mu = offspring[:mu]
    population = np.array([x[0] for x in best_mu])
    sigma = np.array([x[1] for x in best_mu])
    # The best overall is the first after sorting
    best_solution, best_fitness = offspring[0][0], offspring[0][2]
    return population, sigma, best_solution, best_fitness


class ImageOptimizer:
    def __init__(self, img_size=(24, 24), target=None):
        """
        Evolution Strategy-based image optimizer.
        
        :param img_size: (height, width)
        :param target: 2D array representing a target image, or None
        """
        self.img_size = img_size
        self.dim = img_size[0] * img_size[1]
        self.bounds = (0, 1)  # Pixel intensity range
        self.target = target

        # Setup visualization
        plt.ion()
        self.fig, self.ax = plt.subplots()
        # Initialize display with random image
        random_image = np.random.rand(*img_size)
        self.img_display = self.ax.imshow(random_image, cmap='gray', vmin=0, vmax=1)
        plt.title("Evolving Image")

    def update_display(self, image_vector):
        """
        Update the displayed image.
        
        :param image_vector: 1D array representing the image
        """
        img = image_vector.reshape(self.img_size)
        self.img_display.set_data(img)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def evolve(self, mu=30, lambd=200, sigma_init=0.01,
               sigma_bound=0.001, max_generations=100000):
        """
        Run the evolutionary strategy to evolve an image.
        
        :param mu: number of parents
        :param lambd: number of offspring
        :param sigma_init: initial step size
        :param sigma_bound: min allowable step size
        :param max_generations: number of generations to run
        :return: best image as a 2D array (shape = img_size)
        """
        # Initialize population & step sizes
        population, sigma = initialize_population(mu, self.img_size,
                                                  self.bounds, sigma_init)
        best_solution, best_fitness = None, float('inf')
        lr = 1 / np.sqrt(self.dim)  # typical choice in self-adaptation

        for gen in range(max_generations):
            # Generate offspring
            offspring = generate_offspring(
                population, sigma, mu, lambd, self.dim, 
                self.bounds, lr, sigma_bound, 
                lambda x: image_fitness(x, self.target)
            )

            # Select survivors
            population, sigma, new_best, new_fitness = select_survivors(offspring, mu)

            # Update best solution if there's improvement
            if new_fitness < best_fitness:
                best_solution, best_fitness = new_best, new_fitness
                self.update_display(best_solution)

            print(f"Gen {gen}: Best Fitness {best_fitness:.6f}")

        return best_solution.reshape(self.img_size)


if __name__ == "__main__":
    # Optionally create a target image
    target = np.zeros((24, 24))
    target[8:16, 8:16] = 1  # White square in the center

    fig, ax = plt.subplots()
    ax.imshow(target, cmap='gray', vmin=0, vmax=1)
    plt.title("Target Image")

    # Create and run the image optimizer
    optimizer = ImageOptimizer(img_size=(24, 24), target=target)
    final_image = optimizer.evolve(max_generations=1000)

    # Show final result
    plt.ioff()
    plt.imshow(final_image, cmap='gray', vmin=0, vmax=1)
    plt.title("Final Evolved Image")
    plt.show()
