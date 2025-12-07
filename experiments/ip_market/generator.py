import random
from typing import Dict, List, Tuple

from .models import Firm, Module
from .prompts import firm_prompt, adversarial_seller_prompt


def generate_firms(n: int, adversarial_fraction: float = 0.0, seed: int = 0) -> List[Firm]:
    rng = random.Random(seed)
    firms: List[Firm] = []
    for i in range(n):
        fid = f"F{i+1}"
        if rng.random() < adversarial_fraction:
            prompt = adversarial_seller_prompt(fid)
        else:
            prompt = firm_prompt(fid)
        firms.append(Firm(firm_id=fid, system_prompt=prompt))
    return firms


def generate_modules(n_firms: int, n_modules: int, seed: int) -> Tuple[List[Module], Dict[str, Dict[str, int]]]:
    rng = random.Random(seed)
    modules: List[Module] = []
    need_levels: Dict[str, Dict[str, int]] = {f"F{i+1}": {} for i in range(n_firms)}
    for m in range(n_modules):
        tech_id = f"T{m+1}"
        owner = f"F{rng.randint(1, n_firms)}"
        quality = rng.choice(["High", "Medium", "Low"])
        integration_costs = {f"F{i+1}": max(0.5, rng.gauss(2.0, 1.0)) for i in range(n_firms)}
        internal_costs = {f"F{i+1}": max(1.0, rng.gauss(6.0, 2.0)) for i in range(n_firms)}
        benefits = {f"F{i+1}": max(0.0, rng.gauss(10.0, 4.0)) for i in range(n_firms)}
        needs = {}
        for i in range(n_firms):
            level = rng.choices([0, 1, 2], weights=[0.4, 0.35, 0.25])[0]
            needs[f"F{i+1}"] = level
            need_levels[f"F{i+1}"][tech_id] = level
        modules.append(
            Module(
                tech_id=tech_id,
                owner=owner,
                quality=quality,
                integration_costs=integration_costs,
                internal_costs=internal_costs,
                benefits=benefits,
                needs=needs,
            )
        )
    return modules, need_levels
