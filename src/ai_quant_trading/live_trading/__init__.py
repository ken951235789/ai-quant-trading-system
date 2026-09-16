"""實盤交易 API 模組。"""

from typing import TYPE_CHECKING

from ai_quant_trading.live_trading.binance_client import (
    BinanceExecutionUnknown,
    BinancePrivateApiError,
    BinancePrivateClient,
)
from ai_quant_trading.live_trading.config import LiveTradingConfig
from ai_quant_trading.live_trading.capital_flows import (
    FlowAdjustedEquity,
    flow_adjusted_equity,
    record_capital_flow,
)
from ai_quant_trading.live_trading.binance_futures_client import (
    BinanceFuturesPrivateClient,
)
from ai_quant_trading.live_trading.credentials import (
    BinanceCredentials,
    credentials_available,
    load_binance_credentials,
)
from ai_quant_trading.live_trading.factory import build_live_gateway
from ai_quant_trading.live_trading.gateway import (
    LiveTradingGateway,
    OrderPreview,
    OrderSubmission,
    PortfolioSnapshot,
)
from ai_quant_trading.live_trading.futures_gateway import (
    FuturesPortfolioSnapshot,
    FuturesProtectionSubmission,
    FuturesTradingGateway,
)
from ai_quant_trading.live_trading.audit import AuditVerification, verify_audit_log
from ai_quant_trading.live_trading.operations import (
    OperationalRiskAssessment,
    OperationalRiskLimits,
    assess_operational_risk,
)
from ai_quant_trading.live_trading.readiness import (
    OperationalEvidence,
    TestnetEvidence,
    assess_operational_evidence,
    assess_testnet_evidence,
)
from ai_quant_trading.live_trading.reporting import (
    DailyTradingReport,
    build_daily_trading_report,
    save_daily_trading_report,
)

if TYPE_CHECKING:
    from ai_quant_trading.live_trading.service import (
        LiveCycleResult,
        LiveSignal,
        ManagedPositionReconciliation,
    )


_SERVICE_EXPORTS = {
    "LiveCycleResult",
    "LiveSignal",
    "ManagedPositionReconciliation",
    "reconcile_managed_spot_position",
    "run_live_signal_cycle",
}

_RL_SERVICE_EXPORTS = {"download_latest_rl_market_data", "run_live_rl_cycle"}


def __getattr__(name: str):
    """延後載入模型服務，避免桌面版啟動時提早初始化原生模型套件。"""
    if name in _RL_SERVICE_EXPORTS:
        from ai_quant_trading.live_trading import rl_service

        value = getattr(rl_service, name)
        globals()[name] = value
        return value
    if name not in _SERVICE_EXPORTS:
        raise AttributeError(name)
    from ai_quant_trading.live_trading import service

    value = getattr(service, name)
    globals()[name] = value
    return value

__all__ = [
    "BinanceCredentials",
    "BinanceExecutionUnknown",
    "BinancePrivateApiError",
    "BinancePrivateClient",
    "BinanceFuturesPrivateClient",
    "AuditVerification",
    "DailyTradingReport",
    "FlowAdjustedEquity",
    "LiveCycleResult",
    "LiveSignal",
    "ManagedPositionReconciliation",
    "LiveTradingConfig",
    "LiveTradingGateway",
    "FuturesTradingGateway",
    "FuturesPortfolioSnapshot",
    "FuturesProtectionSubmission",
    "OrderPreview",
    "OrderSubmission",
    "OperationalEvidence",
    "OperationalRiskAssessment",
    "OperationalRiskLimits",
    "PortfolioSnapshot",
    "TestnetEvidence",
    "assess_operational_evidence",
    "assess_operational_risk",
    "assess_testnet_evidence",
    "build_live_gateway",
    "credentials_available",
    "download_latest_rl_market_data",
    "flow_adjusted_equity",
    "load_binance_credentials",
    "reconcile_managed_spot_position",
    "record_capital_flow",
    "run_live_signal_cycle",
    "run_live_rl_cycle",
    "build_daily_trading_report",
    "save_daily_trading_report",
    "verify_audit_log",
]
