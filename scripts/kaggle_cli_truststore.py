"""讓 Kaggle CLI 使用 Windows 系統憑證存放區。"""

from __future__ import annotations

import truststore


# 某些 Windows Python 發行版無法只靠 certifi 驗證 Kaggle 憑證。
truststore.inject_into_ssl()

from kaggle.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
