import asyncio
import gc
import logging
import tempfile
import uuid
import warnings
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from docling.datamodel.base_models import InputFormat

from docling_jobkit.convert.manager import DoclingConverterManager
from docling_jobkit.datamodel.chunking import BaseChunkerOptions, ChunkingExportOptions
from docling_jobkit.datamodel.convert import ConvertDocumentsOptions
from docling_jobkit.datamodel.result import DoclingTaskResult
from docling_jobkit.datamodel.task import Task, TaskSource, TaskTarget
from docling_jobkit.datamodel.task_meta import TaskType
from docling_jobkit.orchestrators.base_orchestrator import (
    BaseOrchestrator,
)
from docling_jobkit.orchestrators.local.worker import AsyncLocalWorker

_log = logging.getLogger(__name__)


class LocalOrchestratorConfig(BaseModel):
    num_workers: int = 2
    shared_models: bool = False
    scratch_dir: Optional[Path] = None


class LocalOrchestrator(BaseOrchestrator):
    def __init__(
        self,
        config: LocalOrchestratorConfig,
        converter_manager: DoclingConverterManager,
    ):
        super().__init__()
        self.config = config
        self.task_queue: asyncio.Queue[str] = asyncio.Queue()
        self.queue_list: list[str] = []
        self.cm = converter_manager
        self._task_results: dict[str, DoclingTaskResult] = {}
        self.scratch_dir = self.config.scratch_dir or Path(
            tempfile.mkdtemp(prefix="docling_")
        )

    async def enqueue(
        self,
        sources: list[TaskSource],
        target: TaskTarget,
        task_type: TaskType = TaskType.CONVERT,
        options: ConvertDocumentsOptions | None = None,
        convert_options: ConvertDocumentsOptions | None = None,
        chunking_options: BaseChunkerOptions | None = None,
        chunking_export_options: ChunkingExportOptions | None = None,
    ) -> Task:
        if options is not None and convert_options is None:
            convert_options = options
            warnings.warn(
                "'options' is deprecated and will be removed in a future version. "
                "Use 'conversion_options' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        task_id = str(uuid.uuid4())
        chunking_export_options = chunking_export_options or ChunkingExportOptions()
        task = Task(
            task_id=task_id,
            task_type=task_type,
            sources=sources,
            convert_options=convert_options,
            chunking_options=chunking_options,
            chunking_export_options=chunking_export_options,
            target=target,
        )
        await self.init_task_tracking(task)

        self.queue_list.append(task_id)
        await self.task_queue.put(task_id)
        return task

    async def queue_size(self) -> int:
        return self.task_queue.qsize()

    async def get_queue_position(self, task_id: str) -> Optional[int]:
        return (
            self.queue_list.index(task_id) + 1 if task_id in self.queue_list else None
        )

    async def task_result(
        self,
        task_id: str,
    ) -> Optional[DoclingTaskResult]:
        if task_id not in self._task_results:
            return None
        return self._task_results[task_id]

    async def process_queue(self):
        # Create a pool of workers
        workers = []
        for i in range(self.config.num_workers):
            _log.debug(f"Starting worker {i}")
            w = AsyncLocalWorker(
                i,
                self,
                use_shared_manager=self.config.shared_models,
                scratch_dir=self.scratch_dir,
            )
            worker_task = asyncio.create_task(w.loop())
            workers.append(worker_task)

        # Wait for all workers to complete (they won't, as they run indefinitely)
        await asyncio.gather(*workers)
        _log.debug("All workers completed.")

    async def warm_up_caches(self):
        # Converter with default options
        pdf_format_option = self.cm.get_pdf_pipeline_opts(ConvertDocumentsOptions())
        converter = self.cm.get_converter(pdf_format_option)
        converter.initialize_pipeline(InputFormat.PDF)

    async def delete_task(self, task_id: str):
        _log.info(f"Deleting result of task {task_id=}")
        if task_id in self._task_results:
            # Clear the result to allow garbage collection
            result = self._task_results.pop(task_id, None)
            if result is not None:
                # Clear any large data structures in the result
                if hasattr(result, 'result') and result.result is not None:
                    if hasattr(result.result, 'content'):
                        result.result.content = None
                del result
        await super().delete_task(task_id)
        
        # Trigger garbage collection after deleting task
        gc.collect()
        
        # Free CUDA memory if many tasks have been deleted
        self._maybe_free_cuda_memory()
    
    def _maybe_free_cuda_memory(self) -> None:
        """Free CUDA memory if PyTorch with CUDA is available."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        except Exception as e:
            _log.debug(f"Error clearing CUDA memory: {e}")

    async def clear_converters(self):
        _log.info("Clearing converters and freeing memory...")
        self.cm.clear_cache()
        
        # Also clear any lingering task results
        old_results_count = len(self._task_results)
        self._task_results.clear()
        _log.info(f"Cleared {old_results_count} cached task results")
        
        # Force full garbage collection
        gc.collect()
        gc.collect()  # Run twice to handle cyclic references
        
        # Free CUDA memory
        self._maybe_free_cuda_memory()
        _log.info("Memory cleanup complete")

    async def check_connection(self):
        pass
