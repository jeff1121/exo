# type: ignore
"""透過 eco 進行叢集生命週期管理。

提供 eco 指令（deploy、stop、start、release、
logs、exec）的 subprocess 包裝與 ClusterInfo dataclass。可重用於整合測試、
bench、eval 與 CI 工作流程。
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import signal
import subprocess
import uuid
from dataclasses import dataclass, field
from enum import Enum

from .client import ExoClient


class Thunderbolt(str, Enum):
    A2A = "a2a"  # 全互連（eco --tb-a2a）
    RING = "ring"  # 環狀拓撲（eco --tb-ring）


class Chip(str, Enum):
    M1 = "M1"
    M1_PRO = "M1 Pro"
    M1_MAX = "M1 Max"
    M1_ULTRA = "M1 Ultra"
    M2 = "M2"
    M2_PRO = "M2 Pro"
    M2_MAX = "M2 Max"
    M2_ULTRA = "M2 Ultra"
    M3 = "M3"
    M3_PRO = "M3 Pro"
    M3_MAX = "M3 Max"
    M3_ULTRA = "M3 Ultra"
    M4 = "M4"
    M4_PRO = "M4 Pro"
    M4_MAX = "M4 Max"
    M4_ULTRA = "M4 Ultra"


logger = logging.getLogger("exo_tools.cluster")

# 設定後會從 GitHub branch/tag 部署，而非本地原始碼（rsync）。
_EXO_REF = os.environ.get("EXO_REF")


@dataclass
class ClusterInfo:
    """保存 `eco start --deploy` 執行結果。"""

    hosts: list[str]
    namespace: str
    api_endpoints: dict[str, str]  # host -> url 對應
    api_url: str  # ExoClient 的主要端點

    primary_host: str = ""
    _host: str = field(init=False, repr=False, default="")
    _port: int = field(init=False, repr=False, default=52415)

    def __post_init__(self) -> None:
        if not self.primary_host:
            self.primary_host = self.hosts[0]
        url = self.api_url.replace("http://", "").replace("https://", "")
        parts = url.split(":")
        self._host = parts[0]
        self._port = int(parts[1]) if len(parts) > 1 else 52415

    def make_client(self, timeout_s: float = 7200.0) -> ExoClient:
        return ExoClient(self._host, self._port, timeout_s=timeout_s)


class EcoSession:
    """管理具唯一使用者與自動清理機制的 eco session。

    使用方式：
        session = EcoSession(user_prefix="test")
        cluster = session.start_deploy(count=2, thunderbolt=True)
        ...
        session.stop_all()  # or let atexit handle it

    此 session 會註冊 atexit 與 signal handlers，確保在正常結束、未捕捉例外、SIGTERM 與 SIGHUP 時都能清理。SIGINT 不會被攔截，讓 KeyboardInterrupt 正常傳遞。
    """

    def __init__(self, user_prefix: str = "test") -> None:
        self._session_id = uuid.uuid4().hex[:8]
        self.user = f"{user_prefix}-{self._session_id}"
        self._env = {**os.environ, "USER": self.user}

        # 註冊清理處理器
        atexit.register(self.stop_all)
        for sig in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, self._signal_handler)

    def _signal_handler(self, signum: int, _frame: object) -> None:
        self.stop_all()
        raise SystemExit(128 + signum)

    def stop_all(self) -> None:
        """停止此 session 的所有叢集並釋放所有保留資源。"""
        with contextlib.suppress(Exception):
            subprocess.run(
                ["eco", "stop"],
                capture_output=True,
                text=True,
                timeout=30,
                env=self._env,
            )

    def _run(
        self, args: list[str], *, check: bool = True, timeout: int = 120
    ) -> subprocess.CompletedProcess[str]:
        """以此 session 的使用者身分執行 eco 指令。

        stdout 會被擷取（JSON 輸出），stderr 會直接輸出到
        主控台，讓 eco 進度訊息可見。
        """
        logger.info(f"eco: {' '.join(args)}")
        return subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            check=check,
            timeout=timeout,
            env=self._env,
        )

    def start_deploy(
        self,
        hosts: list[str] | None = None,
        *,
        count: int | None = None,
        thunderbolt: Thunderbolt | None = None,
        chip: Chip | None = None,
        min_memory_gb: float | None = None,
        wait: bool = True,
        ref: str | None = _EXO_REF,
        timeout: int = 600,
    ) -> ClusterInfo:
        """透過 eco 在一組主機上啟動並部署 exo。

        預設會透過 rsync 從本地原始碼部署。設定 EXO_REF
        或傳入 ref= 可改由 GitHub branch/tag 部署（供 CI 使用）。
        """
        cmd: list[str] = ["eco", "--json", "start", "--deploy"]
        if hosts:
            cmd.extend(hosts)
        if count is not None:
            cmd.extend(["--count", str(count)])
        if thunderbolt is not None:
            cmd.append(f"--tb-{thunderbolt.value}")
        if chip is not None:
            cmd.extend(["--chip", chip.value])
        if min_memory_gb is not None:
            cmd.extend(["--min-memory", str(min_memory_gb)])
        if wait:
            cmd.append("--wait")
        if ref:
            cmd.extend(["--ref", ref])

        result = self._run(cmd, timeout=timeout)
        data = json.loads(result.stdout)["data"]
        endpoints: dict[str, str] = data["api_endpoints"]
        primary_host = data["hosts"][0]

        return ClusterInfo(
            hosts=data["hosts"],
            namespace=data["namespace"],
            api_endpoints=endpoints,
            api_url=endpoints[primary_host],
            primary_host=primary_host,
        )

    def stop(self, hosts: list[str], *, keep: bool = False, timeout: int = 120) -> None:
        """停止指定主機上的 exo。若 keep=True，保留資源。"""
        cmd: list[str] = ["eco", "stop"]
        cmd.extend(hosts)
        if keep:
            cmd.append("--keep")
        self._run(cmd, timeout=timeout)

    def start_hosts(
        self, hosts: list[str], *, namespace: str, timeout: int = 300
    ) -> None:
        """將（先前停止的）主機重新加入既有 namespace。"""
        cmd: list[str] = ["eco", "--json", "start"]
        cmd.extend(hosts)
        cmd.extend(["--namespace", namespace])
        self._run(cmd, timeout=timeout)

    def release(self, hosts: list[str], timeout: int = 120) -> None:
        """釋放已保留的主機。"""
        cmd: list[str] = ["eco", "release"]
        cmd.extend(hosts)
        self._run(cmd, timeout=timeout)

    def logs(
        self, hosts: list[str], lines: int = 500, timeout: int = 60
    ) -> dict[str, list[str]]:
        """取得叢集主機最近的日誌。"""
        cmd: list[str] = ["eco", "--json", "logs"]
        cmd.extend(hosts)
        cmd.extend(["-n", str(lines), "--raw"])
        result = self._run(cmd, check=False, timeout=timeout)
        if result.returncode != 0:
            return {"_error": [result.stderr]}
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return {"_raw": result.stdout.splitlines()}

    def exec(self, hosts: list[str], command: str, timeout: int = 120) -> str:
        """透過 eco 在指定主機上執行任意指令。"""
        cmd: list[str] = ["eco", "exec"]
        cmd.extend(hosts)
        cmd.append("--")
        cmd.extend(command.split())
        result = self._run(cmd, check=False, timeout=timeout)
        return result.stdout


def make_client(cluster: ClusterInfo, timeout_s: float = 7200.0) -> ExoClient:
    """由 ClusterInfo 建立 ExoClient。"""
    return cluster.make_client(timeout_s=timeout_s)


def make_client_from_url(url: str, timeout_s: float = 7200.0) -> ExoClient:
    """由 URL 字串（如 `http://host:port`）建立 ExoClient。"""
    url_clean = url.replace("http://", "").replace("https://", "")
    parts = url_clean.split(":")
    host = parts[0]
    port = int(parts[1]) if len(parts) > 1 else 52415
    return ExoClient(host, port, timeout_s=timeout_s)
