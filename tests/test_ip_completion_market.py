from __future__ import annotations

from experiments.ip_completion_market.generator import generate_modules_and_products
from experiments.ip_completion_market.market import (
    build_summary,
    evaluate_launch_attempts,
    product_state_snapshot,
    started_not_finished_products,
)
from experiments.ip_completion_market.models import Product


def test_generate_modules_and_products_is_deterministic() -> None:
    modules_a, products_a = generate_modules_and_products(6, 12, 6, seed=77)
    modules_b, products_b = generate_modules_and_products(6, 12, 6, seed=77)

    assert [(module.tech_id, module.owner, module.quality) for module in modules_a] == [
        (module.tech_id, module.owner, module.quality) for module in modules_b
    ]
    assert [
        (product.product_id, product.target_firm, tuple(product.required_modules), product.launch_bonus)
        for product in products_a
    ] == [
        (product.product_id, product.target_firm, tuple(product.required_modules), product.launch_bonus)
        for product in products_b
    ]


def test_launch_bonus_requires_full_bundle() -> None:
    product = Product(
        product_id="P1",
        target_firm="F1",
        required_modules=["T1", "T2", "T3"],
        launch_bonus=24.0,
    )
    successful, failed, bonus = evaluate_launch_attempts(
        products_by_id={"P1": product},
        holdings_by_firm={"F1": {"T1", "T2", "T3"}},
        launch_attempts_by_firm={"F1": ["P1"]},
        launched_products=set(),
    )

    assert successful == [{"firm_id": "F1", "product_id": "P1", "launch_bonus": 24.0}]
    assert failed == []
    assert bonus == 24.0


def test_partial_bundle_has_no_launch_bonus() -> None:
    product = Product(
        product_id="P1",
        target_firm="F1",
        required_modules=["T1", "T2", "T3"],
        launch_bonus=24.0,
    )
    successful, failed, bonus = evaluate_launch_attempts(
        products_by_id={"P1": product},
        holdings_by_firm={"F1": {"T1", "T2"}},
        launch_attempts_by_firm={"F1": ["P1"]},
        launched_products=set(),
    )

    assert successful == []
    assert failed == [
        {
            "firm_id": "F1",
            "product_id": "P1",
            "reason": "missing_modules",
            "missing_modules": ["T3"],
        }
    ]
    assert bonus == 0.0


def test_started_not_finished_metric_tracks_partial_progress() -> None:
    product = Product(
        product_id="P1",
        target_firm="F1",
        required_modules=["T1", "T2", "T3"],
        launch_bonus=24.0,
    )
    product_state = product_state_snapshot(
        products=[product],
        holdings_by_firm={"F1": {"T1", "T2"}},
        attempted_launches=set(),
        launched_products=set(),
    )
    summary = build_summary(
        products=[product],
        product_state=product_state,
        total_deal_volume=1,
        total_internal_builds=2,
        total_welfare=9.5,
        launch_bonus_captured=0.0,
    )

    assert started_not_finished_products(product_state) == ["P1"]
    assert summary["partial_bundle_rate"] == 1.0
    assert summary["started_not_finished_count"] == 1
    assert summary["launch_bonus_captured"] == 0.0
