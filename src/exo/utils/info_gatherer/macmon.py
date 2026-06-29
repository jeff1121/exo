from typing import Self

from pydantic import BaseModel

from exo.shared.types.profiling import MemoryUsage, SystemPerformanceProfile
from exo.utils.pydantic_ext import TaggedModel


class _TempMetrics(BaseModel, extra="ignore"):
    """此說明已翻譯為繁體中文。"""

    cpu_temp_avg: float
    gpu_temp_avg: float


class _MemoryMetrics(BaseModel, extra="ignore"):
    """此說明已翻譯為繁體中文。"""

    ram_total: int
    ram_usage: int
    swap_total: int
    swap_usage: int


class RawMacmonMetrics(BaseModel, extra="ignore"):
    """此說明已翻譯為繁體中文。

    此說明已翻譯為繁體中文。
    """

    timestamp: str  # 已翻譯註解。
    temp: _TempMetrics
    memory: _MemoryMetrics
    ecpu_usage: tuple[int, float]  # 已翻譯註解。
    pcpu_usage: tuple[int, float]  # 已翻譯註解。
    gpu_usage: tuple[int, float]  # 已翻譯註解。
    all_power: float
    ane_power: float
    cpu_power: float
    gpu_power: float
    gpu_ram_power: float
    ram_power: float
    sys_power: float


class MacmonMetrics(TaggedModel):
    system_profile: SystemPerformanceProfile
    memory: MemoryUsage

    @classmethod
    def from_raw(cls, raw: RawMacmonMetrics) -> Self:
        return cls(
            system_profile=SystemPerformanceProfile(
                gpu_usage=raw.gpu_usage[1],
                temp=raw.temp.gpu_temp_avg,
                sys_power=raw.sys_power,
                pcpu_usage=raw.pcpu_usage[1],
                ecpu_usage=raw.ecpu_usage[1],
            ),
            memory=MemoryUsage.from_bytes(
                ram_total=raw.memory.ram_total,
                ram_available=(raw.memory.ram_total - raw.memory.ram_usage),
                swap_total=raw.memory.swap_total,
                swap_available=(raw.memory.swap_total - raw.memory.swap_usage),
            ),
        )

    @classmethod
    def from_raw_json(cls, json: str) -> Self:
        return cls.from_raw(RawMacmonMetrics.model_validate_json(json))
