from typing import List, Dict, Any
from openai import OpenAI
from .retrieval_strategy import RetrievalStrategy

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

    def answer_question(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Orquesta el flujo: Pregunta -> Búsqueda -> Prompt -> Respuesta
        """
        print(f"🤖 [RAG] Procesando pregunta: '{question}'")
        
        # 1. RETRIEVAL: Conseguir la materia prima (Delegado a la estrategia seleccionada)
        retrieved_docs = self.retriever.retrieve(question, top_k=top_k)

        # Si no hay documentos, cortamos el flujo para no gastar tokens en GPT
        if not retrieved_docs:
            return {
                "answer": "Lo siento, tras analizar la base de datos, no encontré documentos científicos que contengan información relevante para responder a esta pregunta específica.",
                "sources": [],
                "context_used": []
            }

        # 2. CONTEXT BUILDING: Preparar los datos para el LLM
        # GPT necesita saber qué texto viene de qué archivo para poder citar.
        context_text = ""
        unique_sources = set()
        
        for i, doc in enumerate(retrieved_docs):
            # A. Extracción segura de datos (Soporta Dict o Objeto Pydantic)
            if isinstance(doc, dict):
                content = doc.get('content', '')
                raw_source = doc.get('source_file', 'Desconocido')
            else:
                content = getattr(doc, 'content', '')
                raw_source = getattr(doc, 'source_file', 'Desconocido')

            # B. Limpieza del nombre del archivo (Quitar rutas largas de Windows/Linux)
            # De "C:/Users/data/paper_maqui.pdf" a "paper_maqui.pdf"
            filename = raw_source.replace("\\", "/").split("/")[-1]
            unique_sources.add(filename)
            
            # C. Formateo estructurado
            # Le damos etiquetas claras al modelo: [FUENTE X]
            context_text += f"\n--- FRAGMENTO {i+1} (Fuente: {filename}) ---\n"
            context_text += f"{content}\n"

        # 3. PROMPT ENGINEERING: Las reglas del juego
        # Aquí definimos la personalidad y las restricciones.
        system_prompt = (
            "Eres un asistente de investigación experto en fitoquímica, farmacología y alimentos funcionales. "
            "Tu objetivo es responder preguntas basándote EXCLUSIVAMENTE en el contexto proporcionado. "
            "\n\n"
            "REGLAS OBLIGATORIAS:\n"
            "1. NO uses conocimientos previos externos. Si la respuesta no está en el contexto, di 'No cuento con información suficiente en los documentos procesados'.\n"
            "2. Sé preciso y técnico. Usa vocabulario científico (ej: menciona 'capacidad antioxidante', 'polifenoles', 'mecanismo de acción').\n"
            "3. CITA LAS FUENTES: Cuando hagas una afirmación, intenta referenciar el archivo de origen mencionado en el contexto (ej: 'Según el estudio de maqui.pdf...').\n"
            "4. Si hay opiniones contradictorias en los fragmentos, menciónalas."
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