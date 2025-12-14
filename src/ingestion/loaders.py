import re
import uuid
import pymupdf  # fitz
import pymupdf4llm
from typing import List, Dict, Any
from .loaders_interface import AbstractLoader
from src.models import ProcessedChunk

# --- CAMBIO: IMPORTS OPEN SOURCE ---
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings # <--- Nuevo

class MetabolomicsPDFLoader(AbstractLoader):
    """
    Cargador Avanzado Open Source: 
    Usa modelos locales (HuggingFace) para el Chunking Semántico.
    """

    def __init__(self):
        print("-> [Loader] Cargando modelo de embeddings local (esto toma unos segundos la primera vez)...")
        
        # 1. Configurar Modelo Local (Open Source)
        # "all-MiniLM-L6-v2" es rápido y muy bueno para detectar cambios de tema.
        # Si tienes GPU, device='cuda', sino 'cpu'.
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2", 
            model_kwargs={'device': 'cpu'} 
        )
        
        # 2. Configurar el Semantic Chunker
        # percentil 90 suele funcionar bien para modelos locales
        self.semantic_splitter = SemanticChunker(
            self.embedding_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90 
        )

    def load_and_chunk(self, file_path: str) -> List[ProcessedChunk]:
        print(f"-> [SemanticLoader OS] Iniciando: {file_path}")
        
        all_chunks = []
        BATCH_SIZE = 20
        
        try:
            with pymupdf.open(file_path) as doc:
                total_pages = len(doc)
                
                for start_page in range(0, total_pages, BATCH_SIZE):
                    end_page = min(start_page + BATCH_SIZE, total_pages)
                    pages_indices = list(range(start_page, end_page))
                    
                    print(f"   ... Analizando semánticamente págs {start_page} a {end_page} ...")
                    
                    md_text_batch = pymupdf4llm.to_markdown(doc, pages=pages_indices)
                    batch_chunks = self._process_text_batch(md_text_batch, file_path, start_page)
                    all_chunks.extend(batch_chunks)
                    
        except Exception as e:
            print(f"❌ Error crítico leyendo PDF: {e}")

        return all_chunks

    def _process_text_batch(self, text: str, file_path: str, page_offset: int) -> List[ProcessedChunk]:
        chunks = []
        
        # A. Datos Duros (Tablas) - Se mantiene igual
        metabolites = self._extract_metabolites_regex(text)
        for item in metabolites:
            chunks.append(ProcessedChunk(
                chunk_id=str(uuid.uuid4()),
                content=f"Identified metabolite: {item['name']} with m/z {item['mz']} and RT {item['rt']} min.",
                source_file=file_path,
                metadata={
                    "type": "table_row",
                    "mz": item['mz'], 
                    "rt": item['rt'],
                    "compound_name": item['name'],
                    "page_range_start": page_offset
                }
            ))

        # B. Texto Narrativo - CHUNKING SEMÁNTICO LOCAL
        clean_text = re.sub(r"\|.*\|", "", text) 
        sections = clean_text.split("##")
        
        for section in sections:
            section = section.strip()
            if len(section) < 50: continue

            try:
                # El modelo local calcula las distancias aquí
                semantic_docs = self.semantic_splitter.create_documents([section])
                
                for doc in semantic_docs:
                    final_texts = self._safe_fallback_split(doc.page_content)
                    
                    for final_text in final_texts:
                        chunks.append(ProcessedChunk(
                            chunk_id=str(uuid.uuid4()),
                            content=final_text,
                            source_file=file_path,
                            metadata={
                                "type": "narrative_text",
                                "split_strategy": "semantic-local", # Marcamos que fue local
                                "page_range_start": page_offset
                            }
                        ))
            except Exception as e:
                print(f"⚠️ Falló chunking semántico local: {e}")
                fallback_texts = self._safe_fallback_split(section)
                for t in fallback_texts:
                    chunks.append(ProcessedChunk(
                        chunk_id=str(uuid.uuid4()),
                        content=t,
                        source_file=file_path,
                        metadata={"type": "narrative_text_fallback"}
                    ))
                
        return chunks

    def _safe_fallback_split(self, text: str, max_chars=2000) -> List[str]:
        # (Este método se mantiene IDÉNTICO al anterior)
        if len(text) <= max_chars:
            return [text]
        chunks = []
        while len(text) > max_chars:
            split_idx = text.rfind('\n', 0, max_chars)
            if split_idx == -1: split_idx = text.rfind(' ', 0, max_chars)
            if split_idx == -1: split_idx = max_chars
            chunk = text[:split_idx].strip()
            if chunk: chunks.append(chunk)
            text = text[split_idx:].strip()
        if text: chunks.append(text)
        return chunks

    def _extract_metabolites_regex(self, text: str) -> List[Dict[str, Any]]:
        # (Este método se mantiene IDÉNTICO al anterior)
        metabolites = []
        lines = text.split('\n')
        for line in lines:
            if "|" in line and "---" not in line: 
                try:
                    numbers = re.findall(r"(\d+\.\d+)", line)
                    if len(numbers) >= 1:
                        nums_float = [float(n) for n in numbers]
                        mz_candidate = max(nums_float)
                        rt_candidate = min(nums_float) if len(nums_float) > 1 else 0.0
                        if 50 < mz_candidate < 2000:
                            clean_name = re.sub(r"[\d\.\+\-\|]", "", line).strip()
                            clean_name = clean_name.replace("  ", " ")
                            if len(clean_name) > 3:
                                metabolites.append({
                                    "name": clean_name[:50],
                                    "mz": mz_candidate,
                                    "rt": rt_candidate
                                })
                except Exception:
                    continue
        return metabolites