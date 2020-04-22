import os

from typing import List, Tuple, Any
from nptyping import NDArray

from PIL import Image
from random import randint, random
import numpy as np

from multiprocessing import Pool
from itertools import repeat

# Program Types
Colour = NDArray[(3), np.uint8]
Colours = List[Colour]
Size = Tuple[int, int]
Pixels = NDArray[(Any, Any, 3), Colour]

# Program Parameters
results_filepath = 'results/'
src_filepath: str = 'mario.jpg'
src = Image.open(src_filepath)
src_pixels: Pixels = np.array(src)
colours: Colours = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
size: Size = src.size
(width, height) = size
mutation_chance: float = 0.01
mating_pool: int = 100
parent_selection: int = int(np.sqrt(mating_pool))

# Performance Parameters
reporting_frequency: int = 10
processes: int = 16
target_loss: int = 50000

# Parameter Assertions
assert os.path.exists(results_filepath)
assert os.path.exists(src_filepath)
assert size == src.size
assert mutation_chance <= 1
assert np.sqrt(mating_pool).is_integer()
assert mating_pool % parent_selection == 0
assert target_loss >= 0


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

    def calculate_gene_distance(self) -> np.uint16:
        sum_loss: float = 0
        for w in range(width):
            for h in range(height):
                r_loss = (
                    int(src_pixels[w][h][0]) -
                    int(self.pixels[w][h][0])
                )
                g_loss = (
                    int(src_pixels[w][h][1]) -
                    int(self.pixels[w][h][1])
                )
                b_loss = (
                    int(src_pixels[w][h][2]) -
                    int(self.pixels[w][h][2])
                )
                loss = np.sqrt(
                    (r_loss ** 2) +
                    (g_loss ** 2) +
                    (b_loss) ** 2
                )
                sum_loss += loss
        return sum_loss


def pixel_selection(father_pixels: Colour, mother_pixels: Colour):
    mutated_gene = np.zeros((3), dtype=np.uint8)
    if random() <= mutation_chance:
        for n in range(0, 3):
            mutated_gene[n] = randint(0, 255)
        return np.array(mutated_gene, dtype=np.uint8)
    else:
        r = random()
        if r >= 0.5:
            return father_pixels
        elif r <= 0.5:
            return mother_pixels


def mate_genes(father: Gene, mothers: List[Gene]) -> List[Gene]:
    offspring_genes: List[Gene] = []
    for mother in mothers:
        parent_pixels = [father.pixels, mother.pixels]
        pixels = np.zeros((width, height, 3), dtype=np.uint8)
        for w in range(width):
            for h in range(height):
                pixels[w][h] = pixel_selection(
                    parent_pixels[0][w][h],
                    parent_pixels[1][w][h]
                )
                offspring_genes.append(Gene(pixels))
    return offspring_genes


def record_generation(parents: List[Gene], generation: int) -> None:
    best_gene = parents[0]
    loss = best_gene.loss
    pixels = best_gene.pixels
    result = Image.fromarray(pixels)
    result.save(f'results/result-gen-{generation}.png')
    print(f'generation {generation}, loss {loss}')
    return


def create_gene(_):
    return Gene()


# Program Execution
if __name__ == '__main__':
    pool: type(Pool) = Pool(processes)

    genes: List[Gene] = pool.map(
        create_gene,
        range(mating_pool)
    )
    genes.sort(key=lambda gene: gene.loss)
    loss: float = genes[0].loss

    generation: int = 0
    while loss > target_loss:
        if ((generation) % reporting_frequency == 0) or (generation == 0):
            record_generation(genes, generation)

        selected_parents: List[Gene] = [
            genes[n] for n in range(parent_selection)
        ]
        offspring: List[List[Gene]] = pool.starmap(
            mate_genes, zip(
                selected_parents,
                repeat(selected_parents, parent_selection)
            )
        )
        offspring: List[Gene] = [i for j in offspring for i in j]
        offspring.sort(key=lambda gene: gene.loss)
        if offspring[0].loss < selected_parents[0].loss:
            genes = offspring

        generation += 1

    pool.close()
    pool.join()

    record_generation(genes, generation)
