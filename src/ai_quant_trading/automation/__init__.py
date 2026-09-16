"""模擬與實盤共用的無人值守排程工具。"""

from ai_quant_trading.automation.runner import (
    AutomationConfig,
    AutomationPaths,
    AutomationStatus,
    automation_paths,
    read_automation_status,
    request_automation_stop,
    run_automation,
)

__all__ = [
    "AutomationConfig",
    "AutomationPaths",
    "AutomationStatus",
    "automation_paths",
    "read_automation_status",
    "request_automation_stop",
    "run_automation",
]
