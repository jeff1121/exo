1. EXO_BOOTSTRAP_PEERS 目前有問題

4. 我想看到已剖析的網路延遲／頻寬。
5. 我想看到每條連線正在使用多少頻寬。
7. 解決 continuous batching 的問題：當新 prompt 進來時，會在 prefill 完成前阻塞目前 batch 的 decode。
8. 我們希望使用者可以在完全不讓 EXO 連上網路的情況下，將模型複製到新裝置。目前 EXO 需要至少一次網路連線來快取一些檔案，以檢查下載是否完成。相反地，我們應該只要檢查本機是否有非空的模型資料夾，且沒有 .partial 檔案。這表示模型已完整下載並可載入。
13. 記憶體壓力，而不是記憶體使用量。
14. 在 UI 中顯示每條連線的類型（TB5、Ethernet 等）。參考舊版 exo: https://github.com/exo-explore/exo/blob/56f783b38dc6b08ce606b07a5386dc40dae00330/exo/helpers.py#L251
16. 當高優先級連線可用時，動態切換過去。可能需要把 InstanceReplacedAtomically 帶回來。
17. 透過從叢集中其他裝置串流模型來加快模型載入。
18. 新增支援，在測試中指定要使用的網路連線類型。依賴 15/16。
27. 日誌清理 - 依模組過濾日誌，並預設為 DEBUG 日誌等級
28. 在資訊蒐集器中用 ibv_devinfo 驗證 RDMA 連線
