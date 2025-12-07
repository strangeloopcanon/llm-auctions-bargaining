import math
import random
from typing import Dict, List, Tuple

from .models import DepartmentView, Feature

DEPTS = ["Marketing", "Sales", "Product", "Engineering"]


def generate_features(n: int, seed: int) -> List[Feature]:
    rng = random.Random(seed)
    features: List[Feature] = []
    risk_tags = ["normal", "tech_debt_risk", "security_sensitive", "operational_burden"]
    for i in range(n):
        fid = f"F{i+1}"
        title = rng.choice(
            [
                "Self-serve onboarding",
                "Audit log API",
                "Advanced dashboards",
                "Role-based access",
                "AI assistant",
                "Billing revamp",
                "Mobile offline mode",
                "Performance tuning",
                "Compliance pack",
                "Workflow automation",
            ]
        )
        description = rng.choice(
            [
                "Improves first-time user experience.",
                "Enterprise requirement for risk/logging.",
                "Adds insights for power users.",
                "Security and permission hardening.",
                "Adds generative help inside the app.",
                "Overhauls billing accuracy and invoicing.",
                "Offline capability for mobile field teams.",
                "Reduces latency and error budgets.",
                "Adds SOC2/GDPR helpers.",
                "Lets users build custom automations.",
            ]
        )
        eng_cost = rng.randint(1, 8)
        # Correlated values: Marketing & Sales align often; Eng sometimes negative
        base = rng.gauss(0, 1)
        gt_values: Dict[str, float] = {}
        gt_values["Marketing"] = 5 + base + rng.gauss(0, 1)
        gt_values["Sales"] = 5 + base + rng.gauss(0, 1)
        gt_values["Product"] = 5 + rng.gauss(0, 1)
        gt_values["Engineering"] = rng.gauss(0, 1) - math.log1p(eng_cost)
        risk_flag = rng.choices(risk_tags, weights=[0.5, 0.2, 0.2, 0.1], k=1)[0]
        features.append(
            Feature(
                feature_id=fid,
                title=title,
                description=description,
                eng_cost=eng_cost,
                ground_truth_values=gt_values,
                risk_flag=risk_flag,
            )
        )
    return features


def qualitative_label(value: float) -> str:
    if value > 6:
        return "High"
    if value > 4:
        return "Medium"
    return "Low"


def noisy_view(gt: float, rng: random.Random) -> str:
    label = qualitative_label(gt)
    if rng.random() < 0.1:  # noise
        label = rng.choice(["High", "Medium", "Low"])
    return label


def build_department_views(
    features: List[Feature], seed: int
) -> Dict[str, List[DepartmentView]]:
    rng = random.Random(seed)
    views: Dict[str, List[DepartmentView]] = {d: [] for d in DEPTS}
    for f in features:
        for dept in DEPTS:
            label = noisy_view(f.ground_truth_values[dept], rng)
            notes = rng.choice(
                [
                    "Strategic fit",
                    "Customer ask",
                    "Demo friendly",
                    "Tech debt risk",
                    "Operational burden",
                    "Security sensitive",
                    "Competitive gap",
                ]
            )
            # ensure risk flag surfaces to all departments at least in notes
            if f.risk_flag == "tech_debt_risk":
                notes = "Tech debt risk"
            elif f.risk_flag == "security_sensitive":
                notes = "Security sensitive"
            elif f.risk_flag == "operational_burden":
                notes = "Operational burden"
            dept_view = {
                "signal": label,
                "notes": notes,
                "risk_flag": f.risk_flag,
            }
            views[dept].append(
                DepartmentView(
                    feature_id=f.feature_id,
                    title=f.title,
                    eng_cost=f.eng_cost,
                    department_view=dept_view,
                )
            )
    return views
