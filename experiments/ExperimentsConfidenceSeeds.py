import random

import numpy as np

from anomalearn.utils import load_py_json, save_py_json, print_step, print_warning

SEEDS_PATH = "ExperimentsConfidenceSeeds.json"
NUM_SEEDS = 50
MAX_SEED = 10000
MIN_SEED = 0
STEP_SEED = 1

std_population = range(MIN_SEED, MAX_SEED, STEP_SEED)


def save_seeds(seeds_):
    save_py_json(seeds_, SEEDS_PATH)
    print_step(f"New seeds are: {seeds}")
    print_step(f"Seeds saved to {SEEDS_PATH}")


def create_seeds():
    seeds_ = random.sample(std_population, NUM_SEEDS)
    save_seeds(seeds_)


print_step("Trying to load the seeds")
seeds: list | None = load_py_json(SEEDS_PATH)

if seeds is None:
    print_step("Seeds does not exist, they will be created")
    create_seeds()
else:
    print_step("Seeds already exist")

    response = None
    while response is None:
        print_warning("Do you really want to change them? [y/n] ", end="")
        response = input()
        if response.lower() != "n" and response.lower() != "y":
            response = None

    if len(seeds) > NUM_SEEDS and response.lower() == "y":
        response_erase = None
        while response_erase is None:
            print_warning("Seeds are more than NUM_SEEDS. Continue? [y/n] ", end="")
            response_erase = input()
            if response_erase.lower() != "n" and response_erase.lower() != "y":
                response_erase = None

    match response.lower():
        case "y":
            if len(seeds) == NUM_SEEDS:
                print_warning(f"There are already {NUM_SEEDS} seeds. Change NUM_SEED")
            elif len(seeds) > NUM_SEEDS:
                seeds = seeds[:NUM_SEEDS]
                save_seeds(seeds)
            else:
                print_step("Appending new seeds to the loaded ones")
                population = list(set(std_population).difference(seeds))
                new_seeds = random.sample(population, NUM_SEEDS - len(seeds))
                print_step(f"Previous seeds were: {seeds}")
                seeds.extend(new_seeds)
                save_seeds(seeds)
                doubles = np.sum(seeds[1:] == seeds[:-1])
                print_step(f"There are {doubles} duplicate seeds")

        case "n":
            print_step("Seeds will not be modified")
