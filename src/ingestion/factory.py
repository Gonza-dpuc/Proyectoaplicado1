# src/ingestion/factory.py
from .loaders import MetabolomicsPDFLoader

class DocumentLoaderFactory:
    @staticmethod
    def get_loader(file_path: str):
        if file_path.endswith(".pdf"):
            # ¡Aquí usamos nuestra nueva clase Open Source!
            return MetabolomicsPDFLoader() 
        else:
            raise ValueError(f"Formato no soportado: {file_path}")