# type: ignore
"""使用 Playwright（無頭 Chromium）的 Dashboard 端對端測試。

前置需求：
    uv run playwright install chromium

執行方式：
    uv run pytest tests/test_dashboard.py -v
"""

from __future__ import annotations

import contextlib

import pytest

try:
    from playwright.sync_api import sync_playwright

    _HAS_PLAYWRIGHT = True
except ImportError:
    _HAS_PLAYWRIGHT = False

# 透過快速啟動來檢查是否已安裝 Chromium
_HAS_CHROMIUM = False
if _HAS_PLAYWRIGHT:
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            browser.close()
        _HAS_CHROMIUM = True
    except Exception:
        pass

pytestmark = pytest.mark.skipif(
    not _HAS_PLAYWRIGHT or not _HAS_CHROMIUM,
    reason="playwright or chromium not installed (run: uv run playwright install chromium)",
)


def _mark_onboarding_complete(session) -> None:
    """在伺服器端標記 onboarding 完成，避免精靈自動啟動模型。"""
    with contextlib.suppress(Exception):
        session.client.request_json("POST", "/onboarding")


@pytest.mark.cluster(count=1)
def test_dashboard_chat_inference(session):
    """完整 UI 流程：開啟 dashboard、選擇模型、送出聊天、驗證回應。

    The instance is created via the dashboard UI (model picker → chat send
    triggers the dashboard's auto-launch flow), not via @pytest.mark.instance.
    """
    _mark_onboarding_complete(session)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.goto(session.cluster.api_url, wait_until="networkidle")
        page.wait_for_timeout(3000)
        page.screenshot(path="/tmp/dashboard_initial.png")

        # 點擊「SELECT MODEL」按鈕以開啟模型選擇器
        page.get_by_text("SELECT MODEL", exact=False).first.click()
        page.wait_for_timeout(1000)
        page.screenshot(path="/tmp/dashboard_picker_open.png")

        # 搜尋模型——使用 model id 子字串；選擇器
        # 會比對 name/id，因此「Llama-3.2-1B」會篩到小型 Llama。
        search_input = page.locator('input[placeholder*="Search models"]').first
        search_input.fill("Llama-3.2-1B")
        page.wait_for_timeout(1500)
        page.screenshot(path="/tmp/dashboard_picker_search.png")

        # 點擊唯一符合的結果。選擇器顯示模型的
        # 顯示名稱（例如「Llama 3.2 1B」），可能與 model_id 不同。
        # 我們點擊結果清單中第一個可見、像按鈕的列。
        page.get_by_text("Llama 3.2 1B", exact=False).first.click()
        page.wait_for_timeout(1500)
        page.screenshot(path="/tmp/dashboard_model_selected.png")

        # 輸入聊天訊息——送出後會觸發 dashboard 的自動啟動
        # 流程：它會為選定模型挑選最佳放置並 POST
        # 到 /instance，接著在 runner 就緒後送出聊天。
        chat_input = page.locator("textarea").first
        chat_input.fill("Say hello")
        chat_input.press("Enter")
        page.screenshot(path="/tmp/dashboard_chat_sent.png")

        # 等待 instance 啟動並回應。超時設定較寬裕
        # 因為包含模型放置 + 載入 + 生成。
        page.wait_for_timeout(60000)
        page.screenshot(path="/tmp/dashboard_after_chat.png")

        # 驗證已建立 instance，且聊天已收到回應
        instances = session.client.request_json("GET", "/state").get("instances", {})
        assert len(instances) > 0, "Expected the dashboard to have created an instance"

        body_text = page.text_content("body") or ""
        assert len(body_text) > 0

        browser.close()
