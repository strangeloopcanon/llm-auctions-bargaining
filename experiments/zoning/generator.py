import random
from typing import List, Tuple


def generate_world(n_homeowners: int, seed: int) -> Tuple[float, float, List[float]]:
    rng = random.Random(seed)
    V_D = rng.uniform(300_000, 900_000)
    B_R = rng.uniform(100_000, 300_000)
    C_i = [rng.uniform(30_000, 150_000) for _ in range(n_homeowners)]
    return V_D, B_R, C_i
