import json
from typing import List, Any
from openai import OpenAI
from .vector_store_service import VectorStoreService

# ==========================================
# 1. CLASE BASE (Debe ir primero)
# ==========================================
class RetrievalStrategy:
    """Clase base abstracta para estrategias de búsqueda."""
    def __init__(self, vector_service: VectorStoreService, openai_api_key: str):
        self.vector_service = vector_service
        self.client = OpenAI(api_key=openai_api_key)

    def retrieve(self, query: str, top_k: int) -> List[Any]:
        raise NotImplementedError

# ==========================================
# 2. ESTRATEGIAS CONCRETAS
# ==========================================

class HyDERetrieval(RetrievalStrategy):
    """
    HyDE: Genera un documento hipotético y busca vectores similares a él.
    """
    def retrieve(self, query: str, top_k: int = 3) -> List[Any]:
        hypothetical_doc = self._generate_hypothetical_doc(query)
        print(f"👻 HyDE Document: '{hypothetical_doc[:100]}...'")
        
        emb_resp = self.client.embeddings.create(
            input=hypothetical_doc,
            model="text-embedding-3-small"
        )
        query_vector = emb_resp.data[0].embedding
        
        return self.vector_service.query(
            query_vector=query_vector,
            filters=None,
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
    def retrieve(self, query: str, top_k: int = 3) -> List[Any]:
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
                filters=None,
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
    def retrieve(self, query: str, top_k: int = 3) -> List[Any]:
        step_back_query = self._generate_step_back(query)
        print(f"🔙 Step-Back: '{step_back_query}'")
        
        k_split = max(1, top_k // 2)
        results_original = self._search_vector(query, k_split)
        results_step_back = self._search_vector(step_back_query, k_split)
        
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

    def _search_vector(self, text_query: str, k: int):
        emb = self.client.embeddings.create(input=text_query, model="text-embedding-3-small")
        return self.vector_service.query(query_vector=emb.data[0].embedding, filters=None, top_k=k)

# ==========================================
# 3. ENSEMBLE DINÁMICO (Debe ir después de la clase base)
# ==========================================

class CustomEnsembleRetrieval(RetrievalStrategy):
    """
    Recibe una lista de estrategias YA instanciadas y las ejecuta todas.
    """
    def __init__(self, strategies: List[RetrievalStrategy]):
        self.strategies = strategies

    def retrieve(self, query: str, top_k: int = 5) -> List[Any]:
        print(f"⚡ Ensemble con {len(self.strategies)} estrategias...")
        all_results = []
        k_per_strat = max(2, int(top_k * 0.7)) 
        
        for strat in self.strategies:
            try:
                results = strat.retrieve(query, top_k=k_per_strat)
                all_results.extend(results)
            except Exception as e:
                print(f"⚠️ Falló estrategia {type(strat).__name__}: {e}")

        # Deduplicación final
        unique_map = {}
        for doc in all_results:
            c_id = doc.get('chunk_id') if isinstance(doc, dict) else doc.chunk_id
            if c_id not in unique_map:
                unique_map[c_id] = doc
        
        return list(unique_map.values())[:top_k]

# ==========================================
# 4. FACTORY (Al final del archivo)
# ==========================================

def get_strategy_from_selection(selection: List[str], vector_service, api_key) -> RetrievalStrategy:
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
        # Fallback a HyDE si no hay nada seleccionado (o lanzar error)
        return HyDERetrieval(vector_service, api_key)
        
    if len(active_instances) == 1:
        return active_instances[0]
    else:
        return CustomEnsembleRetrieval(strategies=active_instances)