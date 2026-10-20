"""Deprecated generic base for final-dataset online tool workers.

The final-dataset tools now own their processing in each concrete
``<Tool>_worker.py`` module. This file remains only to make stale imports fail
with a clear message instead of silently using a centralized runtime.
"""

from __future__ import annotations

import argparse
import uuid

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.tool_workers.online_workers.final_dataset_tool_specs import get_tool_instruction
from tool_server.utils.server_utils import build_logger


worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"FinalDatasetToolWorker_{worker_id}.log")


class FinalDatasetToolWorker(BaseToolWorker):
    def __init__(
        self,
        tool_name: str,
        controller_addr: str,
        worker_addr: str = "auto",
        worker_id: str = worker_id,
        no_register: bool = False,
        host: str = "0.0.0.0",
        port: int | None = None,
        limit_model_concurrency: int = 8,
        model_semaphore=None,
        wait_timeout: float = 120.0,
        task_timeout: float = 120.0,
    ):
        self.tool_name = tool_name
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            no_register,
            None,
            None,
            tool_name,
            False,
            False,
            "cpu",
            limit_model_concurrency,
            host,
            port,
            model_semaphore,
            wait_timeout,
            task_timeout,
        )

    def init_model(self):
        logger.info("%s worker is ready.", self.tool_name)

    def generate(self, params):
        raise RuntimeError(
            "FinalDatasetToolWorker is deprecated. Use the concrete "
            f"{self.tool_name}_worker.py worker so the tool's own process "
            "function handles the request."
        )

    def get_tool_instruction(self):
        return get_tool_instruction(self.tool_name)


def run_worker(tool_name: str, default_port: int) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--limit-model-concurrency", type=int, default=8)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    worker = FinalDatasetToolWorker(
        tool_name=tool_name,
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        no_register=args.no_register,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
    )
    worker.run()
