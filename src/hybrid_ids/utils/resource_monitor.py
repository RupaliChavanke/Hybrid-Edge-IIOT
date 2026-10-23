"""
Hardware resource monitoring and execution profiling utilities.
Tracks physical memory consumption (RSS), CPU usage, and high-precision timing.
"""

import os
import time
import psutil
from typing import Dict, Any, Generator
from contextlib import contextmanager


class ResourceMonitor:
    """Monitors system memory and CPU during heavy computation tasks."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())

    def get_memory_info(self) -> Dict[str, float]:
        """Returns current process and system memory metrics in megabytes (MB)."""
        mem_info = self.process.memory_info()
        sys_mem = psutil.virtual_memory()
        return {
            "process_rss_mb": round(mem_info.rss / (1024 * 1024), 2),
            "process_vms_mb": round(mem_info.vms / (1024 * 1024), 2),
            "system_available_mb": round(sys_mem.available / (1024 * 1024), 2),
            "system_total_mb": round(sys_mem.total / (1024 * 1024), 2),
            "system_percent_used": sys_mem.percent
        }


@contextmanager
def benchmark_timer() -> Generator[Dict[str, Any], None, None]:
    """Context manager for high-resolution wall-clock and CPU benchmarking."""
    monitor = ResourceMonitor()
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    start_mem = monitor.get_memory_info()["process_rss_mb"]

    result = {}
    try:
        yield result
    finally:
        end_wall = time.perf_counter()
        end_cpu = time.process_time()
        end_mem = monitor.get_memory_info()["process_rss_mb"]

        result["duration_sec"] = round(end_wall - start_wall, 4)
        result["cpu_time_sec"] = round(end_cpu - start_cpu, 4)
        result["start_rss_mb"] = start_mem
        result["end_rss_mb"] = end_mem
        result["delta_rss_mb"] = round(end_mem - start_mem, 2)
