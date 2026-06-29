# 已翻譯註解。
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from exo.api.main import API


def test_http_exception_handler_formats_openai_style() -> None:
    """此說明已翻譯為繁體中文。"""

    app = FastAPI()

    # 設定例外處理器
    api = object.__new__(API)
    api.app = app
    api._setup_exception_handlers()  # 已翻譯註解。

    # 已翻譯註解。
    @app.get("/test-error")
    async def _test_error() -> None:
        raise HTTPException(status_code=500, detail="Test error message")

    @app.get("/test-not-found")
    async def _test_not_found() -> None:
        raise HTTPException(status_code=404, detail="Resource not found")

    client = TestClient(app)

    # 測試 500 錯誤
    response = client.get("/test-error")
    assert response.status_code == 500
    data: dict[str, Any] = response.json()
    assert "error" in data
    assert data["error"]["message"] == "Test error message"
    assert data["error"]["type"] == "Internal Server Error"
    assert data["error"]["code"] == 500

    # 測試 404 錯誤
    response = client.get("/test-not-found")
    assert response.status_code == 404
    data = response.json()
    assert "error" in data
    assert data["error"]["message"] == "Resource not found"
    assert data["error"]["type"] == "Not Found"
    assert data["error"]["code"] == 404
