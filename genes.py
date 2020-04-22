from typing import List, Tuple, Any
from nptyping import NDArray
from PIL import Image
from random import randint, random
import numpy as np
from multiprocessing import Pool

# Program Types
Colour = NDArray[(3), np.uint8]
Colours = List[Colour]
Size = Tuple[int, int]
Pixels = NDArray[(Any, Any, 3), Colour]
LossSum = NDArray[(3), int]

# Program Parameters
src_filepath: str = 'dog.jpg'
src = Image.open(src_filepath)
src_pixels: Pixels = np.array(src)
colours: Colours = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
size: Size = src.size
(width, height) = size
mutation_chance: float = 0.25
generations: int = 10000
pool_size: int = 128
processes: int = 18


class Gene:
    def __init__(self, pixels: Pixels = None):
        if pixels is None:
            self.pixels = self.generate_pixels()
        else:
            self.pixels = pixels
        self.loss = self.calculate_gene_distance()

    def generate_pixels(self) -> Pixels:
        pixels = np.zeros((width, height, 3), dtype=np.uint8)
        for w in range(width):
            for h in range(height):
                pixels[w][h] = self.select_colour()
        return pixels

    def select_colour(self) -> Colour:
        index_value = randint(0, len(colours) - 1)
        return colours[index_value]

    def calculate_gene_distance(self) -> float:
        sum_loss: LossSum = np.zeros((3))
        for w in range(width):
            for h in range(height):
                loss = src_pixels[w][h] - self.pixels[w][h] ** 2
                sum_loss += loss
        return np.sum(sum_loss)


def pixel_selection(father_pixels: Colour, mother_pixels: Colour):
    if random() <= mutation_chance:
        return np.array([randint(0, 255), randint(0, 255), randint(0, 255)])
    else:
        r = random()
        if r >= 0.5:
            return father_pixels
        elif r <= 0.5:
            return mother_pixels


def mate_genes(parents: Tuple[Gene, Gene]) -> Gene:
    (father, mother) = parents
    parent_pixels = [father.pixels, mother.pixels]
    pixels = np.zeros((width, height, 3), dtype=np.uint8)
    for w in range(width):
        for h in range(height):
            pixels[w][h] = pixel_selection(
                parent_pixels[0][w][h],
                parent_pixels[1][w][h]
            )
    return Gene(pixels)


def create_gene(x):
    return Gene()


# Program Execution
if __name__ == '__main__':
    pool = Pool(processes)

    genes: List[Gene] = pool.map(
        create_gene,
        range(pool_size)
    )

    for generation in range(0, generations):
        genes.sort(key=lambda gene: gene.loss)
        genes: List[Gene] = genes[0:10]
        father, mother = genes[randint(0, 3)], genes[randint(0, 3)]

        offspring: List[Gene] = pool.map(
            mate_genes,
            [(father, mother) for i in range(pool_size)]
        )
        offspring.sort(key=lambda gene: gene.loss)
        genes = offspring

        if (generation + 1) % 10 == 0:
            genes.sort(key=lambda gene: gene.loss)
            best_gene = genes[0].pixels
            print(f'generation {generation + 1}, {genes[0].loss}')
            result = Image.fromarray(best_gene)
            result.save(f'results/result-gen-{generation + 1}.png')

    pool.close()
    pool.join()

    genes.sort(key=lambda gene: gene.loss)
    best_gene = genes[0].pixels
    result = Image.fromarray(best_gene)
    result.save('results/result-final.png')
