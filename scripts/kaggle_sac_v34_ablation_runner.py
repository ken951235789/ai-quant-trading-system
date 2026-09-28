"""V3.4 第二階段：因果 cross-fit 與 SAC 三組輸入對照，不授予實盤資格。"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from time import monotonic
import traceback
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def prepare_bundle_root(input_root: Path, destination: Path) -> Path:
    """Kaggle 可能將 Dataset ZIP 自動解開，兩種掛載格式都要接受。"""
    manifests = list(input_root.rglob("formal_manifest.json"))
    if len(manifests) > 1:
        raise ValueError("找到多個訓練包，請只掛載本次指定的私人資料集")
    if manifests:
        root = manifests[0].parent
    else:
        archives = list(input_root.rglob("transformer_v3_formal_bundle.zip"))
        if len(archives) != 1:
            raise FileNotFoundError("找不到唯一的訓練包或 formal_manifest.json")
        destination.mkdir(parents=True, exist_ok=True)
        root = destination.resolve()
        with ZipFile(archives[0]) as archive:
            for member in archive.infolist():
                if not (root / member.filename).resolve().is_relative_to(root):
                    raise ValueError("訓練包包含超出目標資料夾的路徑")
            archive.extractall(root)
    for relative in (
        "formal_manifest.json",
        "data/transformer_v3_formal.csv",
        "src/ai_quant_trading/__init__.py",
    ):
        if not (root / relative).is_file():
            raise FileNotFoundError(f"訓練包缺少必要檔案：{relative}")
    return root


def run_study(source: Path, output: Path, *, smoke: bool = False) -> dict:
    from ai_quant_trading.transformer import TemporalTransformerConfig, TransformerTrainingConfig
    from ai_quant_trading.transformer.crossfit import (
        TransformerCrossFitConfig,
        run_transformer_crossfit,
        load_transformer_crossfit_predictions,
    )
    from ai_quant_trading.transformer.training import train_temporal_transformer
    from ai_quant_trading.reinforcement_learning.ablation import build_sac_observation_ablations
    from ai_quant_trading.reinforcement_learning.config import PortfolioEnvConfig, RLSplitConfig
    from ai_quant_trading.reinforcement_learning.dataset import prepare_rl_dataset
    from ai_quant_trading.reinforcement_learning.storage import save_rl_environment
    from ai_quant_trading.reinforcement_learning.training import train_rl_agent
    from ai_quant_trading.reinforcement_learning.training_config import RLTrainingConfig
    from ai_quant_trading.reinforcement_learning.universal import build_compact_short_term_frame
    from ai_quant_trading.reinforcement_learning.feature_contract import (
        attach_expected_return,
        build_expected_return_contract,
    )

    started = monotonic()
    output.mkdir(parents=True, exist_ok=True)
    if smoke:
        smoke_source = output / "smoke_source.csv"
        pd.read_csv(source, nrows=2500).to_csv(smoke_source, index=False)
        source = smoke_source
    progress_path = output / "sac_v34_progress.json"
    last_messages: dict[str, int] = {}

    def progress(stage: str, payload: dict) -> None:
        percent = int(float(payload.get("progress", 0.0)) * 100)
        write_json(
            progress_path, {**payload, "stage": stage, "elapsed_seconds": monotonic() - started}
        )
        if percent >= last_messages.get(stage, -5) + 5 or payload.get("status") in {
            "complete",
            "failed",
        }:
            print(f"{stage}: {percent}% {payload.get('status', '')}", flush=True)
            last_messages[stage] = percent

    model = TemporalTransformerConfig(
        input_features=256,
        sequence_length=256,
        d_model=128,
        n_heads=8,
        n_layers=4,
        feedforward_dim=384,
        dropout=0.15,
        latent_dim=24,
        return_horizons=(5, 20, 48),
        patch_size=8,
        patch_stride=4,
        hierarchical_direction=True,
        horizon_adapter_dim=64,
    )
    training = TransformerTrainingConfig(
        epochs=20,
        batch_size=96,
        learning_rate=1.2e-4,
        weight_decay=5e-4,
        early_stopping_patience=6,
        device="cuda",
        cpu_threads=4,
        num_workers=2,
        trading_target_mode="terminal_net",
        calibration_fraction=0.5,
        checkpoint_metric="economic_selection_score",
        training_horizon_weights=(0.15, 0.70, 0.15),
        checkpoint_horizon_weights=(0.15, 0.70, 0.15),
        embargo_bars=48,
        recency_half_life_days=365,
        seed=11,
        economic_minimum_trades=30,
        return_loss_weight=0.50,
        volatility_loss_weight=0.20,
        regime_loss_weight=0.10,
        direction_loss_weight=0.75,
        side_loss_weight=0.90,
        quantile_loss_weight=0.15,
        edge_loss_weight=0.90,
        excursion_loss_weight=0.20,
        tradeability_loss_weight=0.75,
        volatility_regime_loss_weight=0.10,
        direction_focal_gamma=1.0,
        direction_class_balance_power=0.25,
        label_smoothing=0.01,
    )
    crossfit = TransformerCrossFitConfig(
        folds=3,
        mode="expanding",
        initial_train_fraction=0.45,
        final_holdout_fraction=0.10,
        minimum_fit_rows=20_000,
        minimum_oos_rows=5_000,
        minimum_final_holdout_rows=5_000,
        expected_return_horizon=20,
        embargo_bars=48,
    )
    if smoke:
        model = replace(
            model,
            input_features=32,
            sequence_length=8,
            d_model=16,
            n_heads=4,
            n_layers=1,
            feedforward_dim=32,
            latent_dim=4,
            patch_size=4,
            patch_stride=2,
            horizon_adapter_dim=8,
            return_horizons=(1, 2),
            timing_horizon=1,
            primary_horizon=2,
            regime_horizon=2,
        )
        training = replace(
            training,
            epochs=1,
            num_workers=0,
            cpu_threads=2,
            device="cpu",
            mixed_precision=False,
            checkpoint_horizon_weights=(0.2, 0.8),
            training_horizon_weights=(0.2, 0.8),
            embargo_bars=2,
            economic_minimum_trades=2,
            drop_constant_features=False,
            max_feature_correlation=None,
        )
        crossfit = replace(
            crossfit,
            folds=2,
            minimum_fit_rows=300,
            minimum_oos_rows=100,
            minimum_final_holdout_rows=100,
            expected_return_horizon=2,
            embargo_bars=2,
        )

    def trainer(path, fold, root, model_config, train_config):
        return train_temporal_transformer(
            [path],
            model_config,
            train_config,
            root,
            progress_callback=lambda event: progress(f"crossfit_{fold.fold}", event),
        ).model_path

    crossfit_result = run_transformer_crossfit(
        source,
        model,
        training,
        crossfit,
        output / "crossfit",
        trainer=trainer,
    )
    oos, _ = load_transformer_crossfit_predictions(crossfit_result.summary_json)
    enriched, columns = build_compact_short_term_frame(oos)
    enriched = attach_expected_return(
        enriched, build_expected_return_contract(int(model.primary_horizon))
    )
    variants = build_sac_observation_ablations(columns)
    environment = PortfolioEnvConfig(
        initial_capital=1000,
        fee_rate=0.0004,
        slippage_rate=0.0002,
        allow_short=True,
        max_position_fraction=0.5,
        max_short_fraction=0.5,
        execution_mode="perpetual",
        leverage=1,
        max_leverage=1,
        max_margin_fraction=0.5,
        normalized_action_space=True,
        action_semantics="hold_close_target",
        expert_kind="short_term",
        episode_length=2016,
        random_start=True,
        risk_per_trade=0.005,
        daily_loss_limit=0.015,
        max_drawdown_limit=0.10,
        max_consecutive_losses=3,
        rebalance_deadband=0.01,
        minimum_holding_bars=4,
        minimum_stop_distance=0.003,
        maximum_stop_distance=0.0075,
        take_profit_distance=0.015,
        max_spread_bps=15,
        max_slippage_bps=10,
        max_atr_fraction=0.05,
        include_position_context=True,
        include_risk_context=True,
        include_trade_plan_context=False,
        minimum_gross_target_cost_multiple=0,
        minimum_net_risk_reward=0,
        drawdown_penalty=0.5,
        turnover_penalty=0,
        downside_penalty=0,
        concentration_penalty=0,
        holding_period_reference=int(model.primary_horizon),
    )
    config = RLTrainingConfig(
        algorithm="sac",
        total_timesteps=100_000,
        device="cpu",
        seed=11,
        n_envs=1,
        cpu_threads=4,
        learning_rate=1e-4,
        gamma=0.995,
        batch_size=256,
        net_arch=(256, 256),
        evaluation_freq=25_000,
        checkpoint_freq=50_000,
        n_eval_episodes=1,
        sac_buffer_size=200_000,
        sac_learning_starts=10_000,
        sac_ent_coef="auto_0.01",
        sac_action_noise="normal",
        sac_action_noise_sigma=0.05,
    )
    if smoke:
        config = replace(
            config,
            total_timesteps=64,
            batch_size=16,
            net_arch=(16, 16),
            cpu_threads=2,
            evaluation_freq=32,
            checkpoint_freq=64,
            sac_buffer_size=200,
            sac_learning_starts=16,
        )
        environment = replace(environment, episode_length=32)
    results = []
    for name, features in variants.items():
        frame = enriched.copy()
        # 對照共用固定風控，Transformer 預測不得從交易計畫或收益閘門繞進控制組。
        if name == "market_only":
            frame = frame.drop(
                columns=[
                    column
                    for column in frame
                    if column.startswith(("transformer_", "u_transformer_"))
                    or column == "expected_return"
                ]
            )
            frame.attrs.pop("expected_return_contract", None)
        dataset = prepare_rl_dataset(
            frame, features, RLSplitConfig(0.60, 0.20, 20 if smoke else 1000)
        )
        artifact = save_rl_environment(
            dataset,
            environment,
            output / name,
            source_path=source,
            transformer_crossfit_summary=crossfit_result.summary_json
            if name != "market_only"
            else None,
        )
        result = train_rl_agent(
            artifact.run_dir,
            config,
            progress_callback=lambda event, stage=name: progress(stage, event),
        )
        reasons = {}
        for split, path in (
            ("validation", result.paths.validation_csv),
            ("test", result.paths.test_csv),
        ):
            evaluation = pd.read_csv(path)
            if "risk_reasons" in evaluation:
                counts = evaluation["risk_reasons"].dropna().astype(str).value_counts()
                reasons[split] = {str(key): int(value) for key, value in counts.items()}
        results.append(
            {
                "variant": name,
                "features": features,
                "metrics": result.metrics,
                "risk_reason_counts": reasons,
                "run_dir": str(result.paths.run_dir),
            }
        )
        write_json(output / "sac_v34_partial.json", {"runs": results, "status": "running"})
    report = {
        "schema_version": 1,
        "status": "complete",
        "study": "three_input_ablations",
        "seeds": [11],
        "steps_per_variant": config.total_timesteps,
        "elapsed_seconds": monotonic() - started,
        "runs": results,
        "source_rows": len(oos),
        "crossfit_summary": str(crossfit_result.summary_json),
        "deployment_allowed": False,
        "limitations": [
            "單 seed 輸入消融，尚未完成多 seed walk-forward 認證",
            "既有歷史資料已參與過研究，封存尾端不宣稱為全新驗收集",
            "cross-fit 成品未綁定單一實盤分析師，僅供研究",
        ],
    }
    write_json(output / "sac_v34_summary.json", report)
    with ZipFile(output / "sac_v34_research_artifacts.zip", "w", ZIP_DEFLATED) as archive:
        for path in output.rglob("*"):
            if (
                path.is_file()
                and (
                    path.suffix in {".json", ".pt", ".zip"}
                    or path.name
                    in {
                        "validation_evaluation.csv",
                        "test_evaluation.csv",
                        "history.csv",
                        "progress.csv",
                        "validation_signal_diagnostics.csv",
                        "test_signal_diagnostics.csv",
                        "validation_fixed_hold_trades.csv",
                        "test_fixed_hold_trades.csv",
                    }
                )
                and path.name != "sac_v34_research_artifacts.zip"
            ):
                archive.write(path, path.relative_to(output))
    progress("complete", {"status": "complete", "progress": 1.0})
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/sac_v34"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.source:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        source = args.source
    else:
        root = prepare_bundle_root(Path("/kaggle/input"), Path("/tmp/ai_quant_v34_sac"))
        sys.path.insert(0, str(root / "src"))
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--no-index",
                "--find-links",
                str(root / "wheelhouse"),
                "stable-baselines3==2.7.1",
                "gymnasium==1.2.3",
            ],
            check=True,
        )
        source = root / "data" / "transformer_v3_formal.csv"
    try:
        run_study(source, args.output, smoke=args.smoke)
        return 0
    except Exception:
        write_json(
            args.output / "sac_v34_summary.json",
            {"status": "failed", "traceback": traceback.format_exc()},
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
