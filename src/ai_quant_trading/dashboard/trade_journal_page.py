"""在既有交易紀錄頁檢視本機圖文日誌，不重跑交易或模型。"""

from pathlib import Path
import json
import sqlite3
import zipfile

import pandas as pd
import streamlit as st

from ai_quant_trading.trading.journal import TradeJournal
from ai_quant_trading.trading.journal_archive import archive_journal, read_asset


def render_trade_journal(root: Path):
    journal = TradeJournal(root)
    try:
        rows = journal.records(limit=200)
    except (OSError, sqlite3.Error):
        st.warning("交易日誌暫時無法讀取；請稍後重試。交易狀態請以帳戶紀錄為準。")
        return
    if (root / "last_error.json").is_file():
        st.warning("日誌曾發生寫入錯誤，請核對原始成交紀錄；日誌不會阻擋風控平倉。")
    recovery = root / "recovery_status.json"
    try:
        if recovery.is_file() and json.loads(recovery.read_text(encoding="utf-8")).get("pending"):
            st.warning("成交帳務已保存，圖文日誌正在等待恢復。")
        sync_status = root / "sync_status.json"
        if sync_status.is_file() and not json.loads(sync_status.read_text(encoding="utf-8")).get("ok"):
            st.warning("交易所成交日誌尚未同步完成，請以原始帳務為準。")
    except (OSError, ValueError):
        st.warning("日誌恢復狀態暫時無法讀取。")
    if not rows:
        st.info("尚無圖文交易日誌。")
        return
    labels = {"entry": "進場", "scale_in": "加碼", "partial_close": "部分平倉", "close": "平倉", "confirmed_fill": "交易所成交"}
    lookup = {r["id"]: r for r in rows}
    def label(record_id):
        row = lookup[record_id]
        stamp = pd.Timestamp(row["timestamp"]).tz_convert("Asia/Taipei").strftime("%Y-%m-%d %H:%M")
        return f"{stamp} | {labels[row['kind']]} | {row['record']['entry']['symbol']}"
    # 使用持久成交 ID，新增紀錄時不讓使用者正在看的交易隨索引漂移。
    selected = st.selectbox("交易日誌（台北時間）", list(lookup), format_func=label,
        key=f"journal_select_{root.parent.name}")
    item = lookup[selected]
    outcome = item["record"]["outcome"]
    if not item["record"].get("chart_coverage_complete", True):
        st.warning("持倉期間行情不完整；圖片僅供部分路徑檢視。")
    if outcome:
        a, b, c = st.columns(3)
        net = outcome.get("net_pnl")
        a.metric("淨損益", f"{net:+.4f}" if net is not None else "待對帳")
        rr, net_r = outcome["planned_reward_risk"], outcome["realized_net_r"]
        b.metric("預定報酬／風險", f"{rr:.2f}:1" if rr is not None else "未定義")
        c.metric("實現 R", f"{net_r:+.3f}R" if net_r is not None else "未定義")
    try:
        image = read_asset(root, item["id"])
    except (OSError, ValueError, KeyError):
        image = None
        st.error("封存圖片驗證失敗，請保留檔案供核對。")
    if image:
        st.image(image, width="stretch")
    else:
        st.warning("圖片待產生" + (f"：{item['error']}" if item["error"] else ""))
    st.download_button("下載紀錄", json.dumps(item["record"], ensure_ascii=False, indent=2),
        file_name=f"trade_{item['id'][:12]}.json", mime="application/json", icon=":material/download:",
        key=f"journal_download_{root.parent.name}")
    with st.expander("決策與成本明細"):
        st.json(item["record"], expanded=False)
    with st.expander("日誌維護"):
        keep = st.number_input("保留近期散檔筆數", 20, 10000, 200, 20, key=f"journal_keep_{root}")
        if st.button("封存舊圖文", icon=":material/archive:", key=f"journal_archive_{root}"):
            try:
                result = archive_journal(journal, keep_recent=keep, limit=100, prune=True)
                st.success(f"已封存 {result['archived']} 筆")
            except (OSError, ValueError, KeyError, zipfile.BadZipFile):
                st.error("封存未完成；未通過校驗的原檔會保留。")
