import json
from typing import List, Any, Dict, Optional
from collections import defaultdict
from openai import OpenAI
from .vector_store_service import VectorStoreService

# Intentamos importar sentence-transformers para Reranking real
try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

# ==========================================
# 1. CLASE BASE (Debe ir primero)
# ==========================================
class RetrievalStrategy:
    """Clase base abstracta para estrategias de búsqueda."""
    def __init__(self, vector_service: VectorStoreService, openai_api_key: str):
        self.vector_service = vector_service
        self.client = OpenAI(api_key=openai_api_key)

    def retrieve(self, query: str, top_k: int, filters: Optional[Dict[str, Any]] = None) -> List[Any]:
        raise NotImplementedError

# ==========================================
# 2. ESTRATEGIAS CONCRETAS
# ==========================================

class SimpleRetrieval(RetrievalStrategy):
    """
    Búsqueda vectorial simple sin pre-procesamiento de query.
    """
    def retrieve(self, query: str, top_k: int = 3, filters: Optional[Dict[str, Any]] = None) -> List[Any]:
        print(f"🔍 Simple Retrieval: '{query}'")
        emb_resp = self.client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        return self.vector_service.query(
            query_vector=emb_resp.data[0].embedding,
            filters=filters,
            top_k=top_k
        )

class HyDERetrieval(RetrievalStrategy):
    """
    HyDE: Genera un documento hipotético y busca vectores similares a él.
    """
    def retrieve(self, query: str, top_k: int = 3, filters: Optional[Dict[str, Any]] = None) -> List[Any]:
        hypothetical_doc = self._generate_hypothetical_doc(query)
        print(f"👻 HyDE Document: '{hypothetical_doc[:100]}...'")
        
        emb_resp = self.client.embeddings.create(
            input=hypothetical_doc,
            model="text-embedding-3-small"
        )
        query_vector = emb_resp.data[0].embedding
        
        return self.vector_service.query(
            query_vector=query_vector,
            filters=filters, # Propagar filtros
            top_k=top_k
        )

    def _generate_hypothetical_doc(self, query: str) -> str:
        prompt = f"Escribe un párrafo científico breve que responda: {query}"
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content
        except:
            return query

class DecompositionRetrieval(RetrievalStrategy):
    """
    Decomposition: Rompe preguntas complejas en sub-preguntas.
    """
    def retrieve(self, query: str, top_k: int = 3, filters: Optional[Dict[str, Any]] = None) -> List[Any]:
        sub_queries = self._decompose_query(query)
        print(f"🧩 Sub-queries: {sub_queries}")
        
        all_results = []
        seen_ids = set()
        
        # Dividimos el top_k entre las sub-preguntas
        k_per_query = max(2, top_k // len(sub_queries) + 1)

        for sub_q in sub_queries:
            emb_resp = self.client.embeddings.create(
                input=sub_q, model="text-embedding-3-small"
            )
            results = self.vector_service.query(
                query_vector=emb_resp.data[0].embedding,
                filters=filters, # Propagar filtros (Opcional: podría ser selectivo)
                top_k=k_per_query
            )
            for chunk in results:
                # Soporte para dict o objeto
                c_id = chunk.get('chunk_id') if isinstance(chunk, dict) else chunk.chunk_id
                if c_id not in seen_ids:
                    seen_ids.add(c_id)
                    all_results.append(chunk)
        
        return all_results

    def _decompose_query(self, original_query: str) -> List[str]:
        system_prompt = "Descompón la pregunta del usuario en una lista JSON de sub-consultas simples bajo la clave 'queries'."
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Pregunta: {original_query}"}
                ],
                temperature=0
            )
            return json.loads(response.choices[0].message.content).get("queries", [original_query])
        except:
            return [original_query]

class StepBackRetrieval(RetrievalStrategy):
    """
    Step-Back: Genera una pregunta más abstracta para contexto teórico.
    """
    def retrieve(self, query: str, top_k: int = 3, filters: Optional[Dict[str, Any]] = None) -> List[Any]:
        step_back_query = self._generate_step_back(query)
        print(f"🔙 Step-Back: '{step_back_query}'")
        
        k_split = max(1, top_k // 2)
        results_original = self._search_vector(query, k_split, filters)
        results_step_back = self._search_vector(step_back_query, k_split, filters)
        
        # Unir listas (la deduplicación final la hace el ensemble o el servicio)
        return results_original + results_step_back

    def _generate_step_back(self, query: str) -> str:
        prompt = f"Genera una pregunta 'step-back' muy general y teórica basada en: {query}"
        try:
            resp = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return resp.choices[0].message.content
        except:
            return query

    def _search_vector(self, text_query: str, k: int, filters=None):
        emb = self.client.embeddings.create(input=text_query, model="text-embedding-3-small")
        return self.vector_service.query(query_vector=emb.data[0].embedding, filters=filters, top_k=k)

# ==========================================
# 3. ENSEMBLE DINÁMICO (Debe ir después de la clase base)
# ==========================================

class CustomEnsembleRetrieval(RetrievalStrategy):
    """
    Recibe una lista de estrategias YA instanciadas y las ejecuta todas.
    """
    def __init__(self, strategies: List[RetrievalStrategy], use_rrf: bool = True):
        self.strategies = strategies
        self.use_rrf = use_rrf
        # Exponer vector_service del primer hijo para compatibilidad con RerankedRetrieval
        if strategies:
            self.vector_service = strategies[0].vector_service

    def retrieve(self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Any]:
        if not self.use_rrf:
            print(f"⚡ Ensemble con {len(self.strategies)} estrategias (Simple Merge)...")
            all_results = []
            seen_ids = set()
            
            # Si no usamos RRF, traemos top_k de cada una y unimos
            for strat in self.strategies:
                try:
                    results = strat.retrieve(query, top_k=top_k, filters=filters)
                    for doc in results:
                        c_id = doc.get('chunk_id') if isinstance(doc, dict) else doc.chunk_id
                        if c_id not in seen_ids:
                            seen_ids.add(c_id)
                            all_results.append(doc)
                except Exception as e:
                    print(f"⚠️ Falló estrategia {type(strat).__name__}: {e}")
            
            return all_results[:top_k]

        # --- LÓGICA RRF (Por defecto) ---
        print(f"⚡ Ensemble con {len(self.strategies)} estrategias (RRF Fusion)...")
        
        # Diccionario para RRF: chunk_id -> score acumulado
        rrf_scores = defaultdict(float)
        # Mapa para recuperar el objeto documento completo: chunk_id -> doc
        docs_map = {}
        
        # Traemos más candidatos por estrategia para que el reranking sea efectivo
        k_per_strat = max(2, int(top_k * 1.5)) 
        k_rrf_constant = 60 # Constante de suavizado estándar para RRF
        
        for strat in self.strategies:
            try:
                results = strat.retrieve(query, top_k=k_per_strat, filters=filters)
                
                for rank, doc in enumerate(results):
                    # Identificación robusta del ID (soporta dict u objeto)
                    c_id = doc.get('chunk_id') if isinstance(doc, dict) else doc.chunk_id
                    
                    if c_id not in docs_map:
                        docs_map[c_id] = doc
                    
                    # Fórmula RRF: 1 / (k + rank)
                    # Premia documentos que aparecen en múltiples estrategias y en posiciones altas
                    rrf_scores[c_id] += 1 / (k_rrf_constant + rank + 1)
                    
            except Exception as e:
                print(f"⚠️ Falló estrategia {type(strat).__name__}: {e}")

        # Ordenar por score RRF descendente
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Retornar los top_k documentos ganadores
        final_results = [docs_map[cid] for cid in sorted_ids[:top_k]]
        
        return final_results

# ==========================================
# 4. RERANKER (Cross-Encoder)
# ==========================================

class RerankedRetrieval(RetrievalStrategy):
    """
    Decorador que aplica un Cross-Encoder para reordenar los resultados de cualquier estrategia base.
    Implementa el patrón: Retrieve (Top-N) -> Rerank -> Top-K.
    """
    def __init__(self, base_strategy: RetrievalStrategy, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # Inicializamos sin API Key de OpenAI porque el CrossEncoder es local
        super().__init__(base_strategy.vector_service, "dummy")
        self.base_strategy = base_strategy
        self.model = None
        if CrossEncoder:
            try:
                print(f"⚖️ Cargando modelo Cross-Encoder: {model_name}...")
                self.model = CrossEncoder(model_name)
            except Exception as e:
                print(f"⚠️ Error cargando CrossEncoder: {e}")
        else:
            print("⚠️ sentence-transformers no instalado. Reranking desactivado.")

    def retrieve(self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Any]:
        # 1. Recuperación Ampliada (Fetch): Traemos más candidatos (ej: 5x lo solicitado)
        fetch_k = top_k * 5
        candidates = self.base_strategy.retrieve(query, top_k=fetch_k, filters=filters)
        
        if not self.model or not candidates:
            return candidates[:top_k]

        # 2. Preparar pares (Query, Documento) para el modelo
        pairs = []
        for doc in candidates:
            # Extracción segura del contenido
            content = doc.get('content', '') if isinstance(doc, dict) else getattr(doc, 'content', '')
            pairs.append([query, content])

        # 3. Scoring y Reordenamiento
        scores = self.model.predict(pairs)
        
        # Unimos doc con score, ordenamos descendente y cortamos
        scored_candidates = list(zip(candidates, scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        print(f"🔥 Reranking aplicado a {len(candidates)} docs. Top score: {scored_candidates[0][1]:.4f}")
        
        # Devolvemos solo los documentos (sin el score extra)
        return [doc for doc, score in scored_candidates[:top_k]]

# ==========================================
# 5. FACTORY
# ==========================================

def get_strategy_from_selection(selection: List[str], vector_service, api_key, use_reranker: bool = False, use_rrf: bool = True) -> RetrievalStrategy:
    """
    Convierte nombres ["HyDE", "Step-Back"] en objetos de estrategia.
    """
    strategies_map = {
        "HyDE": HyDERetrieval,
        "Decomposition": DecompositionRetrieval,
        "Step-Back": StepBackRetrieval
    }
    
    active_instances = []
    for name in selection:
        if name in strategies_map:
            active_instances.append(strategies_map[name](vector_service, api_key))
            
    if not active_instances:
        # Fallback a Búsqueda Simple si no hay nada seleccionado
        strategy = SimpleRetrieval(vector_service, api_key)
    elif len(active_instances) == 1:
        strategy = active_instances[0]
    else:
        strategy = CustomEnsembleRetrieval(strategies=active_instances, use_rrf=use_rrf)
        
    # Aplicar Reranking si se solicita
    if use_reranker:
        return RerankedRetrieval(base_strategy=strategy)
        
    return strategy