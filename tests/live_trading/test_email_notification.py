from email.message import EmailMessage
from pathlib import Path
import smtplib

import pytest

from ai_quant_trading.live_trading import notification
from ai_quant_trading.live_trading.notification import (
    EmailNotificationSettings,
    email_notification_status,
    notify_event,
    run_email_failure_recovery_drill,
    send_test_email,
)


class FakeSMTP:
    instances: list["FakeSMTP"] = []

    def __init__(self, host: str, port: int, **kwargs: object) -> None:
        self.host = host
        self.port = port
        self.kwargs = kwargs
        self.calls: list[object] = []
        self.message: EmailMessage | None = None
        self.__class__.instances.append(self)

    def __enter__(self) -> "FakeSMTP":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def ehlo(self) -> None:
        self.calls.append("ehlo")

    def starttls(self, **kwargs: object) -> None:
        self.calls.append(("starttls", kwargs))

    def login(self, username: str, password: str) -> None:
        self.calls.append(("login", username, password))

    def send_message(self, message: EmailMessage) -> None:
        self.calls.append("send_message")
        self.message = message


def _set_email_environment(monkeypatch: pytest.MonkeyPatch, *, security: str = "ssl") -> None:
    values = {
        "AI_QUANT_EMAIL_ENABLED": "YES",
        "AI_QUANT_EMAIL_SMTP_HOST": "smtp.example.com",
        "AI_QUANT_EMAIL_SMTP_PORT": "465" if security == "ssl" else "587",
        "AI_QUANT_EMAIL_SECURITY": security,
        "AI_QUANT_EMAIL_USERNAME": "robot@example.com",
        "AI_QUANT_EMAIL_PASSWORD": "app-password",
        "AI_QUANT_EMAIL_FROM": "robot@example.com",
        "AI_QUANT_EMAIL_TO": "owner@example.com,backup@example.com",
        "AI_QUANT_EMAIL_LEVELS": "error,critical",
        "AI_QUANT_EMAIL_NOTIFY_RECOVERY": "YES",
        "AI_QUANT_EMAIL_SUBJECT_PREFIX": "[測試系統]",
        "AI_QUANT_EMAIL_TIMEOUT_SECONDS": "10",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_email_status_never_exposes_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_email_environment(monkeypatch)

    status = email_notification_status(tmp_path / "missing.env")

    assert status["configured"] is True
    assert status["recipient_count"] == 2
    assert "username" not in status
    assert "password" not in status
    assert "owner@example.com" not in str(status)
    assert "app-password" not in repr(EmailNotificationSettings.from_environment(tmp_path))


def test_ssl_test_email_uses_tls_and_writes_local_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_email_environment(monkeypatch)
    FakeSMTP.instances.clear()
    monkeypatch.setattr(notification.smtplib, "SMTP_SSL", FakeSMTP)
    monkeypatch.setattr(notification.ssl, "create_default_context", lambda: object())
    target = tmp_path / "notifications.log"

    send_test_email(target, tmp_path / "missing.env")

    smtp = FakeSMTP.instances[-1]
    assert smtp.host == "smtp.example.com"
    assert smtp.port == 465
    assert ("login", "robot@example.com", "app-password") in smtp.calls
    assert smtp.message is not None
    assert smtp.message["To"] == "owner@example.com, backup@example.com"
    assert "實盤系統告警" in str(smtp.message["Subject"])
    assert "Email 告警測試寄送成功" in target.read_text(encoding="utf-8")


def test_starttls_email_upgrades_connection_before_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = EmailNotificationSettings(
        enabled=True,
        smtp_host="smtp.example.com",
        smtp_port=587,
        security="starttls",
        username="robot@example.com",
        password="app-password",
        sender="robot@example.com",
        recipients=("owner@example.com",),
    )
    FakeSMTP.instances.clear()
    monkeypatch.setattr(notification.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(notification.ssl, "create_default_context", lambda: object())

    notification._send_email(settings, "critical", "測試事件")

    calls = FakeSMTP.instances[-1].calls
    assert calls[0] == "ehlo"
    assert calls[1][0] == "starttls"
    assert calls[2] == "ehlo"
    assert calls[3] == ("login", "robot@example.com", "app-password")


def test_notification_email_failure_never_interrupts_trading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_email_environment(monkeypatch)
    monkeypatch.setattr(
        notification,
        "_send_email",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(smtplib.SMTPException("拒絕")),
    )
    target = tmp_path / "notifications.log"

    notify_event(
        target,
        "critical",
        "api_secret=do-not-log 交易所對帳失敗",
        env_path=tmp_path / "missing.env",
    )

    content = target.read_text(encoding="utf-8")
    assert "do-not-log" not in content
    assert "交易所對帳失敗" in content
    assert "Email 通知被拒絕或失敗：SMTPException" in content


def test_warning_is_opt_in_but_watchdog_recovery_is_sent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_email_environment(monkeypatch)
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        notification,
        "_send_email",
        lambda _settings, level, message: sent.append((level, message)),
    )
    target = tmp_path / "notifications.log"

    notify_event(target, "warning", "一般警告", env_path=tmp_path / "missing.env")
    notify_event(target, "info", "Watchdog 恢復正常", env_path=tmp_path / "missing.env")

    assert sent == [("info", "Watchdog 恢復正常")]


def test_invalid_or_unencrypted_email_settings_are_rejected() -> None:
    settings = EmailNotificationSettings(
        enabled=True,
        smtp_host="smtp.example.com\r\nBcc:attacker.example",
        smtp_port=25,
        security="plain",  # type: ignore[arg-type]
        username="robot@example.com",
        password="app-password",
        sender="robot@example.com",
        recipients=("owner@example.com\r\nBcc:attacker@example.com",),
    )

    errors = settings.validation_errors()

    assert any("SMTP 主機" in item for item in errors)
    assert any("加密模式" in item for item in errors)
    assert any("收件者" in item for item in errors)


def test_failure_recovery_drill_sends_two_messages_without_trading_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_email_environment(monkeypatch)
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        notification,
        "_send_email",
        lambda _settings, level, message: sent.append((level, message)),
    )
    target = tmp_path / "notifications.log"

    messages = run_email_failure_recovery_drill(target, tmp_path / "missing.env")

    assert [level for level, _message in sent] == ["critical", "info"]
    assert "心跳逾時" in messages[0]
    assert "恢復正常" in messages[1]
    assert not (tmp_path / "emergency_halt.json").exists()
    content = target.read_text(encoding="utf-8")
    assert "故障演練通知寄送成功" in content
    assert "恢復演練通知寄送成功" in content
