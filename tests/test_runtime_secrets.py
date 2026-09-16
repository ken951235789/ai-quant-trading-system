"""原始碼版與封裝版秘密檔路徑測試。"""

from pathlib import Path

from ai_quant_trading.runtime_secrets import runtime_env_path


def test_source_runtime_keeps_project_env(tmp_path: Path) -> None:
    assert runtime_env_path(tmp_path, frozen=False) == tmp_path.resolve() / ".env"


def test_packaged_runtime_uses_user_config_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = tmp_path / "private-config"
    install = tmp_path / "shared-app"
    monkeypatch.setenv("AI_QUANT_CONFIG_DIR", str(config))

    result = runtime_env_path(install, frozen=True)

    assert result == config.resolve() / ".env"
    assert not result.is_relative_to(install.resolve())
