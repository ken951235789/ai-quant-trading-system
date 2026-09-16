"""強化學習參數手冊完整性測試。"""

from dataclasses import fields
from pathlib import Path

from ai_quant_trading.reinforcement_learning import (
    PortfolioEnvConfig,
    RLSplitConfig,
    RLTrainingConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMETER_GUIDE = PROJECT_ROOT / "docs" / "強化學習全參數教學與硬體建議.md"


def test_parameter_guide_covers_every_config_field() -> None:
    """新增設定欄位時，要求同步補上中文參數教學。"""
    guide = PARAMETER_GUIDE.read_text(encoding="utf-8")
    config_types = (RLSplitConfig, PortfolioEnvConfig, RLTrainingConfig)

    missing = [
        field.name
        for config_type in config_types
        for field in fields(config_type)
        if f"`{field.name}`" not in guide
    ]

    assert not missing, f"參數手冊缺少說明：{', '.join(missing)}"


def test_parameter_guide_contains_all_recommended_horizons() -> None:
    guide = PARAMETER_GUIDE.read_text(encoding="utf-8")

    for horizon in ("短期", "1 天", "7 天", "1 個月", "長期"):
        assert horizon in guide
