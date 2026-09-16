"""實盤事件的本地、Webhook 與加密 Email 通知。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import parseaddr
import logging
import os
from pathlib import Path
import re
import smtplib
import ssl
from typing import Literal
from urllib.parse import urlsplit

from dotenv import load_dotenv
import requests

from ai_quant_trading.data_collection.http_client import create_secure_session
from ai_quant_trading.runtime_secrets import runtime_env_path


_DEFAULT_WEBHOOK_HOSTS = {"hooks.slack.com", "discord.com", "discordapp.com"}
_EMAIL_LEVELS = {"warning", "error", "critical"}
_SMTP_HOST_PATTERN = re.compile(r"^[A-Za-z0-9.-]+$")
_SENSITIVE_PATTERNS = (
    re.compile(r"(?i)(api[_ -]?(?:key|secret)|password|token|signature)\s*[:=]\s*([^\s,;]+)"),
    re.compile(r"(?i)(postgresql(?:\+psycopg)?://[^:\s]+:)([^@\s]+)(@)"),
)


@dataclass(frozen=True, slots=True)
class EmailNotificationSettings:
    """SMTP 設定；密碼欄位不會出現在 repr 或狀態畫面。"""

    enabled: bool = False
    smtp_host: str = ""
    smtp_port: int = 465
    security: Literal["ssl", "starttls"] = "ssl"
    username: str = ""
    password: str = field(default="", repr=False)
    sender: str = ""
    recipients: tuple[str, ...] = ()
    alert_levels: frozenset[str] = frozenset({"error", "critical"})
    notify_recovery: bool = True
    subject_prefix: str = "[AI Quant]"
    timeout_seconds: float = 10.0

    @classmethod
    def from_environment(
        cls,
        env_path: str | Path | None = None,
    ) -> "EmailNotificationSettings":
        """讀取 `.env`；既有程序環境變數的優先順序較高。"""
        candidate = Path(env_path) if env_path is not None else runtime_env_path()
        if candidate.exists():
            load_dotenv(candidate, override=False)

        levels = frozenset(
            item.strip().lower()
            for item in os.getenv("AI_QUANT_EMAIL_LEVELS", "error,critical").split(",")
            if item.strip()
        )
        recipients = tuple(
            item.strip()
            for item in os.getenv("AI_QUANT_EMAIL_TO", "").split(",")
            if item.strip()
        )
        security = os.getenv("AI_QUANT_EMAIL_SECURITY", "ssl").strip().lower()
        return cls(
            enabled=os.getenv("AI_QUANT_EMAIL_ENABLED", "NO").strip() == "YES",
            smtp_host=os.getenv("AI_QUANT_EMAIL_SMTP_HOST", "").strip(),
            smtp_port=_read_int_environment("AI_QUANT_EMAIL_SMTP_PORT", 465),
            security=security,  # type: ignore[arg-type]
            username=os.getenv("AI_QUANT_EMAIL_USERNAME", "").strip(),
            password=os.getenv("AI_QUANT_EMAIL_PASSWORD", "").strip(),
            sender=os.getenv("AI_QUANT_EMAIL_FROM", "").strip(),
            recipients=recipients,
            alert_levels=levels,
            notify_recovery=(
                os.getenv("AI_QUANT_EMAIL_NOTIFY_RECOVERY", "YES").strip() == "YES"
            ),
            subject_prefix=os.getenv("AI_QUANT_EMAIL_SUBJECT_PREFIX", "[AI Quant]").strip(),
            timeout_seconds=_read_float_environment(
                "AI_QUANT_EMAIL_TIMEOUT_SECONDS",
                10.0,
            ),
        )

    @property
    def effective_sender(self) -> str:
        return self.sender or self.username

    def validation_errors(self) -> tuple[str, ...]:
        """回傳可直接顯示在 UI 的安全設定錯誤，不包含秘密內容。"""
        errors: list[str] = []
        if not self.smtp_host or not _SMTP_HOST_PATTERN.fullmatch(self.smtp_host):
            errors.append("SMTP 主機格式不合法")
        if not 1 <= self.smtp_port <= 65535:
            errors.append("SMTP Port 必須介於 1 到 65535")
        if self.security not in {"ssl", "starttls"}:
            errors.append("Email 加密模式只能是 ssl 或 starttls")
        if not self.username or not self.password:
            errors.append("SMTP 帳號或應用程式密碼尚未設定")
        if not _valid_email_address(self.effective_sender):
            errors.append("寄件者 Email 格式不合法")
        if not self.recipients:
            errors.append("至少要設定一個收件者")
        elif any(not _valid_email_address(item) for item in self.recipients):
            errors.append("收件者 Email 格式不合法")
        if not self.alert_levels or not self.alert_levels <= _EMAIL_LEVELS:
            errors.append("Email 告警層級只能包含 warning、error、critical")
        if "\r" in self.subject_prefix or "\n" in self.subject_prefix:
            errors.append("Email 主旨前綴不可換行")
        if not 1.0 <= self.timeout_seconds <= 60.0:
            errors.append("SMTP 逾時必須介於 1 到 60 秒")
        return tuple(errors)

    @property
    def configured(self) -> bool:
        return self.enabled and not self.validation_errors()


def _read_int_environment(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return 0


def _read_float_environment(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return 0.0


def _valid_email_address(value: str) -> bool:
    if not value or "\r" in value or "\n" in value:
        return False
    _display_name, address = parseaddr(value)
    local, separator, domain = address.rpartition("@")
    return bool(separator and local and "." in domain and " " not in address)


def email_notification_status(
    env_path: str | Path | None = None,
) -> dict[str, object]:
    """提供介面使用的非敏感狀態，不回傳帳號、密碼或完整信箱。"""
    settings = EmailNotificationSettings.from_environment(env_path)
    return {
        "enabled": settings.enabled,
        "configured": settings.configured,
        "smtp_host": settings.smtp_host,
        "smtp_port": settings.smtp_port,
        "security": settings.security,
        "recipient_count": len(settings.recipients),
        "alert_levels": tuple(sorted(settings.alert_levels)),
        "notify_recovery": settings.notify_recovery,
        "errors": settings.validation_errors() if settings.enabled else (),
    }


def redact_sensitive_text(message: str) -> str:
    """遮罩憑證、Token 與資料庫密碼，避免寫入 log 或外部通知。"""
    redacted = str(message)
    redacted = _SENSITIVE_PATTERNS[0].sub(r"\1=***", redacted)
    redacted = _SENSITIVE_PATTERNS[1].sub(r"\1***\3", redacted)
    return redacted[:4000]


def _validated_webhook_url(raw_url: str) -> str:
    parsed = urlsplit(raw_url)
    host = (parsed.hostname or "").lower()
    configured = {
        item.strip().lower()
        for item in os.getenv("AI_QUANT_NOTIFICATION_ALLOWED_HOSTS", "").split(",")
        if item.strip()
    }
    allowed = configured or _DEFAULT_WEBHOOK_HOSTS
    if parsed.scheme != "https" or not host or host not in allowed:
        raise ValueError("Webhook 必須是已允許的 HTTPS 主機")
    if parsed.username or parsed.password:
        raise ValueError("Webhook URL 不可包含帳號密碼")
    return raw_url


def configure_live_logger(log_dir: str | Path) -> logging.Logger:
    """建立不含 API 憑證的輪次與錯誤日誌。"""
    target = Path(log_dir)
    target.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ai_quant_trading.live_trading")
    logger.setLevel(logging.INFO)
    log_path = (target / "live_trading.log").resolve()
    if not any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename).resolve() == log_path
        for handler in logger.handlers
    ):
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger


def notify_local(path: str | Path, level: str, message: str) -> None:
    """把重要事件寫到本地通知檔，未設定外部服務時不傳送任何資料。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with target.open("a", encoding="utf-8") as stream:
        stream.write(f"{timestamp}\t{level.upper()}\t{redact_sensitive_text(message)}\n")


def _send_email(
    settings: EmailNotificationSettings,
    level: str,
    message: str,
) -> None:
    errors = settings.validation_errors()
    if errors:
        raise ValueError("；".join(errors))

    safe_message = redact_sensitive_text(message)
    email = EmailMessage()
    email["Subject"] = f"{settings.subject_prefix} [{level.upper()}] 實盤系統告警"
    email["From"] = parseaddr(settings.effective_sender)[1]
    email["To"] = ", ".join(parseaddr(item)[1] for item in settings.recipients)
    email.set_content(
        "AI Quant Trading System 偵測到事件。\n\n"
        f"UTC 時間：{datetime.now(timezone.utc).isoformat()}\n"
        f"層級：{level.upper()}\n"
        f"內容：{safe_message}\n\n"
        "請登入本機介面並核對交易所帳戶、保護單及緊急停機狀態。"
    )

    context = ssl.create_default_context()
    if settings.security == "ssl":
        with smtplib.SMTP_SSL(
            settings.smtp_host,
            settings.smtp_port,
            timeout=settings.timeout_seconds,
            context=context,
        ) as client:
            client.login(settings.username, settings.password)
            client.send_message(email)
        return

    with smtplib.SMTP(
        settings.smtp_host,
        settings.smtp_port,
        timeout=settings.timeout_seconds,
    ) as client:
        client.ehlo()
        client.starttls(context=context)
        client.ehlo()
        client.login(settings.username, settings.password)
        client.send_message(email)


def send_test_email(
    path: str | Path,
    env_path: str | Path | None = None,
) -> None:
    """由操作介面送出測試信；設定錯誤或傳送失敗時交由介面顯示。"""
    settings = EmailNotificationSettings.from_environment(env_path)
    if not settings.enabled:
        raise ValueError("請先把 AI_QUANT_EMAIL_ENABLED 設為 YES")
    try:
        _send_email(settings, "test", "這是一封實盤告警測試信，沒有送出任何交易。")
    except (OSError, smtplib.SMTPException) as exc:
        raise RuntimeError("SMTP 伺服器拒絕連線、登入或寄送") from exc
    notify_local(path, "info", "Email 告警測試寄送成功")


def run_email_failure_recovery_drill(
    path: str | Path,
    env_path: str | Path | None = None,
) -> tuple[str, str]:
    """寄送一封模擬故障與一封恢復通知，不修改交易或停機狀態。"""
    settings = EmailNotificationSettings.from_environment(env_path)
    if not settings.configured:
        errors = settings.validation_errors()
        raise ValueError("Email 尚未完成設定" + (f"：{'；'.join(errors)}" if errors else ""))
    if not settings.notify_recovery:
        raise ValueError("請先把 AI_QUANT_EMAIL_NOTIFY_RECOVERY 設為 YES")
    messages = (
        "[演練] Testnet Worker 心跳逾時。這是通知演練，沒有送單或停機。",
        "[演練] Watchdog 恢復正常。這是通知演練，沒有送單或解除停機。",
    )
    try:
        _send_email(settings, "critical", messages[0])
        notify_local(path, "info", "Email 故障演練通知寄送成功")
        _send_email(settings, "info", messages[1])
        notify_local(path, "info", "Email 恢復演練通知寄送成功")
    except (OSError, smtplib.SMTPException) as exc:
        notify_local(path, "warning", f"Email 故障演練失敗：{type(exc).__name__}")
        raise RuntimeError("SMTP 伺服器拒絕演練通知") from exc
    return messages


def _should_send_email(
    settings: EmailNotificationSettings,
    level: str,
    message: str,
) -> bool:
    if not settings.enabled:
        return False
    if level in settings.alert_levels:
        return True
    return level == "info" and settings.notify_recovery and "恢復正常" in message


def notify_event(
    path: str | Path,
    level: str,
    message: str,
    *,
    env_path: str | Path | None = None,
) -> None:
    """保存本地通知，並把符合層級的事件送往已設定的外部通道。"""
    normalized_level = level.lower()
    notify_local(path, normalized_level, message)

    try:
        email_settings = EmailNotificationSettings.from_environment(env_path)
        if _should_send_email(email_settings, normalized_level, message):
            _send_email(email_settings, normalized_level, message)
    except (OSError, smtplib.SMTPException, ValueError) as exc:
        notify_local(path, "warning", f"Email 通知被拒絕或失敗：{type(exc).__name__}")

    if normalized_level not in _EMAIL_LEVELS:
        return
    webhook_url = os.getenv("AI_QUANT_NOTIFICATION_WEBHOOK_URL", "").strip()
    if not webhook_url:
        return
    try:
        safe_url = _validated_webhook_url(webhook_url)
        safe_message = redact_sensitive_text(message)
        session = create_secure_session(trust_environment=False)
        response = session.post(
            safe_url,
            json={"content": safe_message, "text": safe_message},
            timeout=10,
            allow_redirects=False,
        )
        if 300 <= response.status_code < 400:
            raise requests.RequestException("Webhook 不允許重新導向")
        response.raise_for_status()
    except (requests.RequestException, ValueError) as exc:
        notify_local(path, "warning", f"外部 Webhook 通知被拒絕或失敗：{type(exc).__name__}")
