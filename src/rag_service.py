from typing import List, Dict, Any
from openai import OpenAI
from .retrieval_strategy import RetrievalStrategy
import re

class RAGService:
    """
    Servicio encargado de la GENERACIÓN (La 'G' de RAG).
    Toma los documentos recuperados, construye un prompt científico y genera la respuesta.
    """

    def __init__(self, retrieval_strategy: RetrievalStrategy, openai_api_key: str):
        self.retriever = retrieval_strategy
        self.client = OpenAI(api_key=openai_api_key)
        # Usamos gpt-3.5-turbo por costo/velocidad, pero para ciencia gpt-4o es mejor.
        self.model_name = "gpt-3.5-turbo" 

    def _extract_filters_from_query(self, query: str) -> Dict[str, Any]:
        """Detecta intenciones de búsqueda estructurada (m/z) en lenguaje natural."""
        filters = {}
        # Regex para capturar m/z con tolerancia
        mz_match = re.search(r"(?:m/z|mz|mass)\s*[:=]?\s*(\d+\.?\d*)", query, re.IGNORECASE)
        
        if mz_match:
            try:
                mz_val = float(mz_match.group(1))
                # Definimos una ventana de tolerancia (ej: ±0.05 Da)
                tolerance = 0.05 
                filters["mz"] = {
                    "gte": mz_val - tolerance,
                    "lte": mz_val + tolerance
                }
                print(f"🧪 [RAG] Filtro Químico Detectado: m/z {mz_val} (±{tolerance})")
            except ValueError:
                pass
        return filters

    def _repack_context(self, docs: List[Any]) -> str:
        """
        Técnica de Repacking: Agrupa fragmentos por documento fuente para 
        darle coherencia al LLM y ahorrar tokens de cabeceras repetidas.
        """
        grouped = {}
        
        for doc in docs:
            # Extracción segura
            if isinstance(doc, dict):
                content = doc.get('content', '')
                src = doc.get('source_file', 'Desconocido')
            else:
                content = getattr(doc, 'content', '')
                src = getattr(doc, 'source_file', 'Desconocido')
            
            # Limpieza nombre
            filename = src.replace("\\", "/").split("/")[-1]
            
            if filename not in grouped:
                grouped[filename] = []
            grouped[filename].append(content)
            
        # Construcción del texto optimizado
        repacked_text = ""
        for i, (filename, fragments) in enumerate(grouped.items()):
            repacked_text += f"\n=== FUENTE {i+1}: {filename} ===\n"
            # Unimos fragmentos del mismo paper con un separador visual
            repacked_text += "\n(...)\n".join(fragments)
            repacked_text += "\n"
            
        return repacked_text

    def answer_question(self, question: str, top_k: int = 5, use_self_query: bool = True, use_repacking: bool = True, generate_response: bool = True) -> Dict[str, Any]:
        """
        Orquesta el flujo: Pregunta -> Búsqueda -> Prompt -> Respuesta
        """
        print(f"🤖 [RAG] Procesando pregunta: '{question}'")
        
        # 1. RETRIEVAL INTELIGENTE: Extraer filtros y buscar
        filters = None
        if use_self_query:
            filters = self._extract_filters_from_query(question)
            
        retrieved_docs = self.retriever.retrieve(question, top_k=top_k, filters=filters)

        # Si no hay documentos, cortamos el flujo para no gastar tokens en GPT
        if not retrieved_docs:
            return {
                "answer": "Lo siento, tras analizar la base de datos, no encontré documentos científicos que contengan información relevante para responder a esta pregunta específica.",
                "sources": [],
                "context_used": []
            }

        # Extraemos fuentes únicas para el reporte final (Lo hacemos antes por si cortamos el flujo)
        unique_sources = set()
        for doc in retrieved_docs:
            if isinstance(doc, dict):
                raw_source = doc.get('source_file', 'Desconocido')
            else:
                raw_source = getattr(doc, 'source_file', 'Desconocido')
            
            filename = raw_source.replace("\\", "/").split("/")[-1]
            unique_sources.add(filename)

        # Si el usuario solo quiere ver los documentos (Modo Debug/Retrieval)
        if not generate_response:
            return {
                "answer": None, # Indicador de que no hubo generación
                "sources": list(unique_sources),
                "context_used": retrieved_docs
            }

        # 2. CONTEXT BUILDING: Preparar los datos para el LLM
        # Usamos Repacking para organizar mejor la información
        if use_repacking:
            context_text = self._repack_context(retrieved_docs)
        else:
            # Modo simple (Concatenación plana)
            context_text = ""
            for i, doc in enumerate(retrieved_docs):
                if isinstance(doc, dict):
                    content = doc.get('content', '')
                    src = doc.get('source_file', 'Desconocido')
                else:
                    content = getattr(doc, 'content', '')
                    src = getattr(doc, 'source_file', 'Desconocido')
                filename = src.replace("\\", "/").split("/")[-1]
                context_text += f"\n--- FRAGMENTO {i+1} (Fuente: {filename}) ---\n{content}\n"

        # 3. PROMPT ENGINEERING: Las reglas del juego
        # Aquí definimos la personalidad y las restricciones.
        system_prompt = (
            "Eres un asistente de investigación experto en fitoquímica, metabolómica y alimentos funcionales. "
            "Tu objetivo es responder a las consultas sintetizando la información del contexto proporcionado de manera narrativa y coherente."
            "\n\n"
            "INSTRUCCIONES DE GENERACIÓN:\n"
            "REGLAS OBLIGATORIAS:\n"
            "1. NO uses conocimientos previos externos. Si la respuesta no está en el contexto, di 'No cuento con información suficiente en los documentos procesados'.\n"
            "2. Sé preciso y técnico. Usa vocabulario científico (ej: menciona 'capacidad antioxidante', 'polifenoles', 'mecanismo de acción').\n"
            "3. CITA LAS FUENTES: Cuando hagas una afirmación, referencia el archivo de origen mencionado en el contexto.\n"
            "4. Si hay opiniones contradictorias en los fragmentos, menciónalas."
            "OTRAS REGLAS:\n   "
            "1. **Estilo Narrativo**: Redacta una respuesta fluida que integre los hallazgos. Evita formatos rígidos o listas desconectadas a menos que sea necesario para la claridad.\n"
            "2. **Contenido**: Si la consulta es sobre una feature química (m/z, RT), explica su posible identificación y bioactividad basándote en la evidencia del contexto. Si es una pregunta teórica, desarrolla una explicación técnica.\n"
            "3. **Uso de Evidencia**: Respalda tus afirmaciones citando las fuentes disponibles en el contexto (ej: 'Según el estudio [Archivo]...').\n"
            "4. **Manejo de Vacíos**: Si el contexto no tiene la respuesta exacta, no digas simplemente 'no hay información'. En su lugar, explica qué información relacionada sí está disponible o resume lo que los documentos mencionan sobre el tema general.\n"
            "5. **Tono**: Científico, preciso y profesional."
        )

        user_prompt = f"Contexto Científico:\n{context_text}\n\nPregunta del Usuario: {question}"

        # 4. GENERATION: Llamada a la IA
        print("🧠 [RAG] Generando respuesta con LLM...")
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3, # Baja temperatura = Más fiel a los datos, menos creativo
                max_tokens=700   # Espacio suficiente para una explicación detallada
            )
            
            final_answer = response.choices[0].message.content
            
        except Exception as e:
            final_answer = f"⚠️ Ocurrió un error generando la respuesta: {e}"

        # 5. RETORNO: Empaquetamos todo para la UI
        return {
            "answer": final_answer,
            "sources": list(unique_sources), # Lista limpia de archivos usados
            "context_used": retrieved_docs   # Para debug si lo necesitamos
        }