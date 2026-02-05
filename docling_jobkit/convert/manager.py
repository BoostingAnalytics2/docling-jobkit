import enum
import gc
import hashlib
import json
import logging
import os
import re
import sys
import threading
from collections.abc import Iterable, Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional, Union

from pydantic import AnyUrl, BaseModel

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.backend.pdf_backend import PdfDocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel import vlm_model_specs
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    GLMOCRTableStructureOptions,
    HunyuanTableStructureOptions,
    OcrOptions,
    PdfBackend,
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
    PictureDescriptionVlmOptions,
    ProcessingPipeline,
    TableFormerMode,
    TableStructureModel,
    TableStructureOptions,
    VlmPipelineOptions,
)
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, InlineVlmOptions
from docling.document_converter import DocumentConverter, FormatOption, PdfFormatOption
from docling.models.factories import get_ocr_factory
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling_core.types.doc import ImageRefMode

from docling_jobkit.datamodel.convert import ConvertDocumentsOptions

_log = logging.getLogger(__name__)


class DoclingConverterManagerConfig(BaseModel):
    artifacts_path: Optional[Path] = None
    options_cache_size: int = 2
    enable_remote_services: bool = False
    allow_external_plugins: bool = False

    max_num_pages: int = sys.maxsize
    max_file_size: int = sys.maxsize

    # Threading pipeline
    queue_max_size: Optional[int] = None
    ocr_batch_size: Optional[int] = None
    layout_batch_size: Optional[int] = None
    table_batch_size: Optional[int] = None
    batch_polling_interval_seconds: Optional[float] = None


# Custom serializer for PdfFormatOption
# (model_dump_json does not work with some classes)
def _hash_pdf_format_option(pdf_format_option: PdfFormatOption) -> bytes:
    data = pdf_format_option.model_dump(serialize_as_any=True)

    # pipeline_options are not fully serialized by model_dump, dedicated pass
    if pdf_format_option.pipeline_options:
        data["pipeline_options"] = pdf_format_option.pipeline_options.model_dump(
            serialize_as_any=True, mode="json"
        )

        # Explicitly serialize table_structure_options to ensure subclass fields are included
        # (Pydantic's serialize_as_any may not propagate to nested models)
        if hasattr(pdf_format_option.pipeline_options, "table_structure_options"):
            tso = pdf_format_option.pipeline_options.table_structure_options
            data["_table_structure_options"] = {
                "type": type(tso).__name__,
                **tso.model_dump(mode="json"),
            }

    # Replace `pipeline_cls` with a string representation
    data["pipeline_cls"] = repr(data["pipeline_cls"])

    # Replace `backend` with a string representation
    data["backend"] = repr(data["backend"])

    # Serialize the dictionary to JSON with sorted keys to have consistent hashes
    serialized_data = json.dumps(data, sort_keys=True)
    options_hash = hashlib.sha1(
        serialized_data.encode(), usedforsecurity=False
    ).digest()

    # Diagnostic logging for table structure options
    if pdf_format_option.pipeline_options and hasattr(
        pdf_format_option.pipeline_options, "table_structure_options"
    ):
        tso = pdf_format_option.pipeline_options.table_structure_options
        _log.info(
            f"_hash_pdf_format_option: table_structure_options type={type(tso).__name__}, "
            f"hash={options_hash.hex()[:12]}"
        )

    return options_hash


def _to_list_of_strings(input_value: Union[str, list[str]]) -> list[str]:
    def split_and_strip(value: str) -> list[str]:
        if re.search(r"[;,]", value):
            return [item.strip() for item in re.split(r"[;,]", value)]
        else:
            return [value.strip()]

    if isinstance(input_value, str):
        return split_and_strip(input_value)
    elif isinstance(input_value, list):
        result = []
        for item in input_value:
            result.extend(split_and_strip(str(item)))
        return result
    else:
        raise ValueError("Invalid input: must be a string or a list of strings.")


class DoclingConverterManager:
    def __init__(self, config: DoclingConverterManagerConfig):
        self.config = config

        self.ocr_factory = get_ocr_factory(
            allow_external_plugins=self.config.allow_external_plugins
        )
        self._options_map: dict[bytes, PdfFormatOption] = {}
        self._get_converter_from_hash = self._create_converter_cache_from_hash(
            cache_size=self.config.options_cache_size
        )

        self._cache_lock = threading.Lock()

    def _create_converter_cache_from_hash(
        self, cache_size: int
    ) -> Callable[[bytes], DocumentConverter]:
        @lru_cache(maxsize=cache_size)
        def _get_converter_from_hash(options_hash: bytes) -> DocumentConverter:
            pdf_format_option = self._options_map[options_hash]
            format_options: dict[InputFormat, FormatOption] = {
                InputFormat.PDF: pdf_format_option,
                InputFormat.IMAGE: pdf_format_option,
            }

            return DocumentConverter(format_options=format_options)

        return _get_converter_from_hash

    def clear_cache(self):
        """Clear all cached converters and free GPU/CPU memory.
        
        This method:
        1. Gets all cached converters before clearing the cache
        2. Clears the initialized pipelines (which hold ML models)
        3. Clears the LRU cache
        4. Runs garbage collection
        5. Frees CUDA memory if available
        """
        with self._cache_lock:
            # Get cache info to find all cached converters
            cache_info = self._get_converter_from_hash.cache_info()
            _log.info(f"Clearing converter cache: {cache_info.currsize} converters cached")
            
            # Clear the initialized pipelines from all cached converters
            # by iterating through the options map and getting each converter
            converters_to_clear = []
            for options_hash in list(self._options_map.keys()):
                try:
                    # Try to get converter from cache (won't create new one if not cached)
                    converter = self._get_converter_from_hash(options_hash)
                    converters_to_clear.append(converter)
                except KeyError:
                    pass
            
            # Clear pipelines from each converter
            for converter in converters_to_clear:
                if hasattr(converter, 'initialized_pipelines'):
                    pipelines = converter.initialized_pipelines
                    _log.info(f"Clearing {len(pipelines)} initialized pipelines")
                    
                    # Try to unload models from each pipeline
                    for cache_key, pipeline in list(pipelines.items()):
                        self._unload_pipeline_models(pipeline)
                    
                    # Clear the pipelines dict
                    pipelines.clear()
            
            # Clear the options map
            self._options_map.clear()
            
            # Clear the LRU cache
            self._get_converter_from_hash.cache_clear()
        
        # Force garbage collection
        gc.collect()
        
        # Free CUDA memory if available
        self._free_cuda_memory()
        
        _log.info("Converter cache cleared and memory freed")
    
    def _unload_pipeline_models(self, pipeline: Any) -> None:
        """Attempt to unload ML models from a pipeline to free memory."""
        model_attrs = [
            'preprocessing_model',
            'ocr_model', 
            'layout_model',
            'table_model',
            'assemble_model',
            'reading_order_model',
            'vlm_model',
            'vlm',
        ]
        
        for attr in model_attrs:
            if hasattr(pipeline, attr):
                model = getattr(pipeline, attr)
                if model is not None:
                    # Try to call unload if available
                    if hasattr(model, 'unload'):
                        try:
                            model.unload()
                        except Exception as e:
                            _log.debug(f"Error unloading {attr}: {e}")
                    # Set to None to allow GC
                    try:
                        setattr(pipeline, attr, None)
                    except Exception:
                        pass
        
        # Clear enrichment pipe if present
        if hasattr(pipeline, 'enrichment_pipe'):
            pipeline.enrichment_pipe = []
    
    def _free_cuda_memory(self) -> None:
        """Free CUDA memory if PyTorch with CUDA is available."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                _log.info("CUDA memory cache cleared")
        except ImportError:
            pass
        except Exception as e:
            _log.debug(f"Error clearing CUDA memory: {e}")

    def get_converter(self, pdf_format_option: PdfFormatOption) -> DocumentConverter:
        with self._cache_lock:
            options_hash = _hash_pdf_format_option(pdf_format_option)
            self._options_map[options_hash] = pdf_format_option
            converter = self._get_converter_from_hash(options_hash)
            cache_info = self._get_converter_from_hash.cache_info()
            _log.info(
                f"get_converter: hash={options_hash.hex()[:12]}, "
                f"cache hits={cache_info.hits}, misses={cache_info.misses}, "
                f"size={cache_info.currsize}/{cache_info.maxsize}"
            )
        return converter

    def _parse_standard_pdf_opts(
        self, request: ConvertDocumentsOptions, artifacts_path: Optional[Path]
    ) -> PdfPipelineOptions:
        try:
            kind = (
                request.ocr_engine.value
                if isinstance(request.ocr_engine, enum.Enum)
                else str(request.ocr_engine)
            )
            ocr_options: OcrOptions = self.ocr_factory.create_options(  # type: ignore
                kind=kind,
                force_full_page_ocr=request.force_ocr,
            )
        except ImportError as err:
            raise ImportError(
                "The requested OCR engine"
                f" (ocr_engine={request.ocr_engine})"
                " is not available on this system. Please choose another OCR engine "
                "or contact your system administrator.\n"
                f"{err}"
            )

        if request.ocr_lang is not None:
            ocr_options.lang = request.ocr_lang

        # Determine table structure options based on selected model
        table_structure_model = getattr(request, 'table_structure_model', TableStructureModel.TABLEFORMER)
        if isinstance(table_structure_model, str):
            table_structure_model = TableStructureModel(table_structure_model)

        if table_structure_model == TableStructureModel.HUNYUAN:
            import os
            hunyuan_url = os.environ.get("DOCLING_HUNYUAN_SERVER_URL", "http://localhost:8000")
            hunyuan_scale = float(os.environ.get("DOCLING_HUNYUAN_SCALE", "3.0"))
            table_structure_options = HunyuanTableStructureOptions(
                server_url=hunyuan_url,
                scale=hunyuan_scale,
                mode=TableFormerMode(request.table_mode),
                do_cell_matching=request.table_cell_matching,
            )
        elif table_structure_model == TableStructureModel.GLM_OCR:
            import os
            glm_ocr_url = os.environ.get("DOCLING_GLM_OCR_SERVER_URL", "http://localhost:8002")
            glm_ocr_scale = float(os.environ.get("DOCLING_GLM_OCR_SCALE", "2.0"))
            table_structure_options = GLMOCRTableStructureOptions(
                server_url=glm_ocr_url,
                scale=glm_ocr_scale,
                mode=TableFormerMode(request.table_mode),
                do_cell_matching=request.table_cell_matching,
            )
        else:
            # Default: TableFormer
            table_structure_options = TableStructureOptions(
                mode=TableFormerMode(request.table_mode),
                do_cell_matching=request.table_cell_matching,
            )

        _log.info(
            f"_parse_standard_pdf_opts: selected table_structure_model={table_structure_model.value}, "
            f"options type={type(table_structure_options).__name__}"
        )

        pipeline_options = PdfPipelineOptions(
            artifacts_path=artifacts_path,
            allow_external_plugins=self.config.allow_external_plugins,
            enable_remote_services=self.config.enable_remote_services,
            document_timeout=request.document_timeout,
            do_ocr=request.do_ocr,
            ocr_options=ocr_options,
            do_table_structure=request.do_table_structure,
            do_code_enrichment=request.do_code_enrichment,
            do_formula_enrichment=request.do_formula_enrichment,
            do_picture_classification=request.do_picture_classification,
            do_picture_description=request.do_picture_description,
            table_structure_options=table_structure_options,
        )

        if request.image_export_mode != ImageRefMode.PLACEHOLDER:
            pipeline_options.generate_page_images = True
            if request.image_export_mode == ImageRefMode.REFERENCED:
                pipeline_options.generate_picture_images = True
            if request.images_scale:
                pipeline_options.images_scale = request.images_scale

        if request.picture_description_local is not None:
            pipeline_options.picture_description_options = (
                PictureDescriptionVlmOptions.model_validate(
                    request.picture_description_local.model_dump()
                )
            )

        if request.picture_description_api is not None:
            pipeline_options.picture_description_options = (
                PictureDescriptionApiOptions.model_validate(
                    request.picture_description_api.model_dump()
                )
            )
        pipeline_options.picture_description_options.picture_area_threshold = (
            request.picture_description_area_threshold
        )

        # Forward the definition of the following attributes, if they are not none
        for attr in (
            "queue_max_size",
            "ocr_batch_size",
            "layout_batch_size",
            "table_batch_size",
            "batch_polling_interval_seconds",
        ):
            if value := getattr(self.config, attr):
                setattr(pipeline_options, attr, value)

        return pipeline_options

    def _parse_backend(
        self, request: ConvertDocumentsOptions
    ) -> type[PdfDocumentBackend]:
        if request.pdf_backend == PdfBackend.DLPARSE_V1:
            backend: type[PdfDocumentBackend] = DoclingParseDocumentBackend
        elif request.pdf_backend == PdfBackend.DLPARSE_V2:
            backend = DoclingParseV2DocumentBackend
        elif request.pdf_backend == PdfBackend.DLPARSE_V4:
            backend = DoclingParseV4DocumentBackend
        elif request.pdf_backend == PdfBackend.PYPDFIUM2:
            backend = PyPdfiumDocumentBackend
        else:
            raise RuntimeError(f"Unexpected PDF backend type {request.pdf_backend}")

        return backend

    def _parse_vlm_pdf_opts(
        self, request: ConvertDocumentsOptions, artifacts_path: Optional[Path]
    ) -> VlmPipelineOptions:
        pipeline_options = VlmPipelineOptions(
            artifacts_path=artifacts_path,
            document_timeout=request.document_timeout,
            enable_remote_services=self.config.enable_remote_services,
        )

        if request.vlm_pipeline_model in (
            None,
            vlm_model_specs.VlmModelType.GRANITEDOCLING,
        ):
            pipeline_options.vlm_options = vlm_model_specs.GRANITEDOCLING_TRANSFORMERS
            if sys.platform == "darwin":
                try:
                    import mlx_vlm  # noqa: F401

                    pipeline_options.vlm_options = vlm_model_specs.GRANITEDOCLING_MLX
                except ImportError:
                    _log.warning(
                        "To run GraniteDocling faster, please install mlx-vlm:\n"
                        "pip install mlx-vlm"
                    )

        elif request.vlm_pipeline_model == vlm_model_specs.VlmModelType.GRANITE_VISION:
            pipeline_options.vlm_options = vlm_model_specs.GRANITE_VISION_TRANSFORMERS

        elif (
            request.vlm_pipeline_model
            == vlm_model_specs.VlmModelType.GRANITE_VISION_OLLAMA
        ):
            pipeline_options.vlm_options = vlm_model_specs.GRANITE_VISION_OLLAMA

        elif request.vlm_pipeline_model == vlm_model_specs.VlmModelType.GLM_OCR:
            glm_ocr_url = os.environ.get(
                "DOCLING_GLM_OCR_SERVER_URL", "http://localhost:8002"
            )
            glm_ocr_opts = vlm_model_specs.GLM_OCR_API.model_copy()
            glm_ocr_opts.url = AnyUrl(f"{glm_ocr_url}/v1/chat/completions")
            glm_ocr_opts.timeout = 600  # Cold-start: proxy needs ~300s to start vLLM backend
            pipeline_options.vlm_options = glm_ocr_opts
            pipeline_options.enable_remote_services = True  # GLM-OCR requires remote API

        if request.vlm_pipeline_model_local is not None:
            pipeline_options.vlm_options = InlineVlmOptions.model_validate(
                request.vlm_pipeline_model_local.model_dump()
            )

        if request.vlm_pipeline_model_api is not None:
            pipeline_options.vlm_options = ApiVlmOptions.model_validate(
                request.vlm_pipeline_model_api.model_dump()
            )

        return pipeline_options

    # Computes the PDF pipeline options and returns the PdfFormatOption and its hash
    def get_pdf_pipeline_opts(
        self,
        request: ConvertDocumentsOptions,
    ) -> PdfFormatOption:
        artifacts_path: Optional[Path] = None
        if self.config.artifacts_path is not None:
            expanded_path = self.config.artifacts_path.expanduser()
            if str(expanded_path.absolute()) == "":
                _log.info(
                    "artifacts_path is an empty path, model weights will be downloaded "
                    "at runtime."
                )
                artifacts_path = None
            elif expanded_path.is_dir():
                _log.info(
                    "artifacts_path is set to a valid directory. "
                    "No model weights will be downloaded at runtime."
                )
                artifacts_path = expanded_path
            else:
                _log.warning(
                    "artifacts_path is set to an invalid directory. "
                    "The system will download the model weights at runtime."
                )
                artifacts_path = None
        else:
            _log.info(
                "artifacts_path is unset. "
                "The system will download the model weights at runtime."
            )

        pipeline_options: Union[PdfPipelineOptions, VlmPipelineOptions]
        if request.pipeline == ProcessingPipeline.STANDARD:
            pipeline_options = self._parse_standard_pdf_opts(request, artifacts_path)
            backend = self._parse_backend(request)
            pdf_format_option = PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=backend,
            )

        elif request.pipeline == ProcessingPipeline.VLM:
            pipeline_options = self._parse_vlm_pdf_opts(request, artifacts_path)
            pdf_format_option = PdfFormatOption(
                pipeline_cls=VlmPipeline, pipeline_options=pipeline_options
            )
        else:
            raise NotImplementedError(
                f"The pipeline {request.pipeline} is not implemented."
            )

        return pdf_format_option

    def convert_documents(
        self,
        sources: Iterable[Union[Path, str, DocumentStream]],
        options: ConvertDocumentsOptions,
        headers: Optional[dict[str, Any]] = None,
    ) -> Iterable[ConversionResult]:
        pdf_format_option = self.get_pdf_pipeline_opts(options)
        converter = self.get_converter(pdf_format_option)
        with self._cache_lock:
            converter.initialize_pipeline(format=InputFormat.PDF)
        results: Iterator[ConversionResult] = converter.convert_all(
            sources,
            headers=headers,
            page_range=options.page_range,
            max_file_size=self.config.max_file_size,
            max_num_pages=self.config.max_num_pages,
        )

        return results
