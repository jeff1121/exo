# Evolution Plan

## 目標
- 全倉庫翻譯為繁體中文（zh-tw），涵蓋文件、README、程式註解與 docstring。
- 執行程式品質檢查（Code Review）並修正可確認問題至乾淨狀態。
- 執行 CodeQL 安全掃描並修正可確認問題至乾淨狀態。
- 產出可直接貼到 PR 訊息的報告摘要。

## 執行步驟
1. 使用多代理平行翻譯，並按路徑分區避免衝突。
2. 檢查翻譯後檔案，確認未改動執行邏輯與 API 字面值。
3. 執行格式化／靜態檢查／測試，記錄既有問題與新增問題。
4. 執行 Code Review + CodeQL（parallel validation）。
5. 修正工具回報中的有效問題，必要時重跑驗證直到乾淨。
6. 彙整最終報告並寫入 PR 訊息：
   - 變更範圍
   - 程式品質檢查結果
   - CodeQL 安全掃描結果
   - 已修正問題與剩餘風險

## PR 訊息草稿區塊（可直接貼上）
### Summary
- 完成全倉庫 zh-tw 翻譯（文件 + 註解/docstring）。
- 已完成品質與安全驗證，並修正有效問題。

### Code Quality Review
- 狀態：待驗證
- 問題數：待填寫
- 修正摘要：待填寫

### CodeQL Security Scan
- 狀態：待驗證
- 問題數：待填寫
- 修正摘要：待填寫

### Residual Risks
- 待填寫
