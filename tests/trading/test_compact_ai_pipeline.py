"""小型 CPU 全流程：新版 Transformer、SAC、持倉重播與即時 observation。"""

import json

import numpy as np
import pandas as pd

from ai_quant_trading.features.market_context import add_model_market_features
from ai_quant_trading.reinforcement_learning import (
    PortfolioEnvConfig,
    PortfolioTradingEnv,
    RLSplitConfig,
    RLTrainingConfig,
    latest_rl_target,
    load_rl_policy,
    prepare_rl_dataset,
    save_rl_environment,
    train_rl_agent,
)
from ai_quant_trading.reinforcement_learning.feature_contract import (
    attach_expected_return,
    build_expected_return_contract,
    select_compact_short_term_features,
)
from ai_quant_trading.reinforcement_learning.policy import prepare_rl_policy_market_frame
from ai_quant_trading.reinforcement_learning.universal import add_universal_rl_features
from ai_quant_trading.transformer import (
    TemporalTransformerConfig,
    TransformerTrainingConfig,
    infer_transformer_context_frame,
    train_temporal_transformer,
    transformer_oos_provenance,
)
from tests.reinforcement_learning.test_feature_contract import compact_market_frame


def test_compact_transformer_sac_pipeline_runs_without_cloud_or_live_api(tmp_path) -> None:
    source_frame = compact_market_frame(rows=800).drop(
        columns=["transformer_return_5", "transformer_return_20", "transformer_available"],
    )
    source_frame = add_model_market_features(source_frame)
    source = tmp_path / "synthetic_btc.csv"
    source_frame.to_csv(source, index=False)
    transformer = train_temporal_transformer(
        [source],
        TemporalTransformerConfig(
            input_features=256,
            sequence_length=8,
            d_model=16,
            n_heads=4,
            n_layers=1,
            feedforward_dim=32,
            latent_dim=4,
            return_horizons=(1, 5, 20),
            patch_size=2,
            patch_stride=1,
            hierarchical_direction=True,
            horizon_adapter_dim=8,
        ),
        TransformerTrainingConfig(
            epochs=1,
            batch_size=64,
            device="cpu",
            mixed_precision=False,
            num_workers=0,
            cpu_threads=1,
            probability_calibration=False,
        ),
        tmp_path / "transformer",
        run_name="compact_contract_smoke",
    )
    provenance = transformer_oos_provenance(transformer.model_path, source_frame)
    inferred = infer_transformer_context_frame(
        transformer.model_path, source_frame, device="cpu", batch_size=64
    )
    timestamps = pd.to_datetime(inferred.timestamp, utc=True)
    inferred = inferred.loc[
        timestamps.gt(pd.Timestamp(provenance["safe_rl_start_exclusive"]))
    ].reset_index(drop=True)
    inferred = attach_expected_return(inferred, build_expected_return_contract(5))
    enriched = add_universal_rl_features(inferred)
    columns = select_compact_short_term_features(enriched)
    assert len(columns) == 112
    dataset = prepare_rl_dataset(enriched, columns, RLSplitConfig(min_rows_per_split=10))
    environment_config = PortfolioEnvConfig(
        allow_short=True,
        max_short_fraction=0.5,
        max_position_fraction=0.5,
        normalized_action_space=True,
        execution_mode="perpetual",
        episode_length=24,
        random_start=True,
        take_profit_distance=0.015,
    )
    artifact = save_rl_environment(
        dataset,
        environment_config,
        tmp_path / "rl",
        source_path=source,
        transformer_checkpoint=transformer.model_path,
        transformer_provenance=provenance,
    )
    training = train_rl_agent(
        artifact.run_dir,
        RLTrainingConfig(
            algorithm="sac",
            total_timesteps=32,
            batch_size=4,
            net_arch=(16, 16),
            checkpoint_freq=16,
            evaluation_freq=16,
            device="cpu",
            cpu_threads=1,
            sac_learning_starts=4,
            sac_buffer_size=100,
            save_replay_buffer=False,
        ),
    )
    metadata = json.loads(training.paths.metadata_json.read_text(encoding="utf-8"))
    assert metadata["environment_ai_context"]["expected_return_contract"]["horizon_bars"] == 5
    policy = load_rl_policy(training.paths.run_dir, device="cpu")
    prepared = prepare_rl_policy_market_frame(source_frame, policy)
    assert prepared.expected_return.iloc[-1] == prepared.transformer_return_5.iloc[-1]
    signal = latest_rl_target(prepared, policy, cash_ratio=1.0, position_ratio=0.0)
    assert np.isfinite(signal.target_fraction)
    assert abs(signal.target_fraction) <= 0.5

    # 只驗證撮合與多空算式能逐根運作，不用煙霧測試的收益宣稱策略有效。
    environment = PortfolioTradingEnv(dataset.test, columns, environment_config)
    observation, _ = environment.reset(seed=11)
    assert observation.shape == policy.model.observation_space.shape
    for index in range(20):
        observation, reward, terminated, truncated, info = environment.step(
            np.array([0.3 if index < 10 else -0.3], dtype=np.float32),
        )
        assert np.isfinite(observation).all()
        assert np.isfinite(reward) and info["equity"] > 0
        assert not terminated and not truncated
