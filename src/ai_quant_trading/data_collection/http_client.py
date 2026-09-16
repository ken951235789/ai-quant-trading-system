"""為市場資料 API 建立不修改全域 SSL 的安全 HTTP Session。"""

from __future__ import annotations

import ssl
import sys
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class NativeTruststoreAdapter(HTTPAdapter):
    """只讓掛載此 adapter 的 HTTPS 請求使用作業系統信任庫。"""

    def __init__(self, ssl_context: ssl.SSLContext, *args: Any, **kwargs: Any) -> None:
        self.ssl_context = ssl_context
        super().__init__(*args, **kwargs)

    def init_poolmanager(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("ssl_context", self.ssl_context)
        super().init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, proxy: str, **proxy_kwargs: Any) -> Any:
        proxy_kwargs.setdefault("ssl_context", self.ssl_context)
        return super().proxy_manager_for(proxy, **proxy_kwargs)


def create_secure_ssl_context() -> ssl.SSLContext:
    """建立可供 requests 與 WebSocket 共用的安全 SSL context。"""
    if sys.platform == "win32":
        try:
            import truststore
        except ImportError:
            return ssl.create_default_context()
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return ssl.create_default_context()


def create_secure_session(*, trust_environment: bool = True) -> requests.Session:
    """建立 HTTPS Session；私有 API 可停用環境 Proxy，避免憑證被轉送。"""
    session = requests.Session()
    session.trust_env = trust_environment
    retries = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD", "OPTIONS"}),
        respect_retry_after_header=True,
    )
    default_adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", default_adapter)
    if sys.platform != "win32":
        session.mount("https://", HTTPAdapter(max_retries=retries))
        return session

    context = create_secure_ssl_context()
    session.mount("https://", NativeTruststoreAdapter(context, max_retries=retries))
    return session


def require_session(session: requests.Session | None) -> requests.Session:
    """把延後建立的 Session 轉成明確前置條件，避免依賴可被移除的 assert。"""
    if session is None:
        raise RuntimeError("HTTP Session 尚未初始化")
    return session
