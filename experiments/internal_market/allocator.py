from typing import Dict, List

from .models import Feature


def select_features_by_ratio(features: List[Feature], support: Dict[str, int], capacity: int) -> List[Feature]:
    enriched = []
    for f in features:
        fid = f.feature_id
        s = support.get(fid, 0)
        c = f.eng_cost
        p = s / c if c > 0 else 0
        enriched.append((p, f))
    enriched.sort(key=lambda x: x[0], reverse=True)
    selected: List[Feature] = []
    used = 0
    for p, f in enriched:
        if p <= 0:
            continue
        if used + f.eng_cost <= capacity:
            selected.append(f)
            used += f.eng_cost
    return selected
