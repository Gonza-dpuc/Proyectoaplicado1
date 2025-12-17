import subprocess
import streamlit as st


def _run_step(title: str, command: str):
    """Ejecuta un comando del pipeline y muestra resultados en Streamlit."""
    with st.status(title, expanded=True) as status:
        try:
            status.write(f"Ejecutando: `{command}` ...")
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                check=True,
            )
            if result.stdout:
                status.write(result.stdout)
            if result.stderr:
                status.write(result.stderr)
            status.update(label=f"{title} — COMPLETADO", state="complete")
        except subprocess.CalledProcessError as e:
            status.update(label=f"{title} — ERROR", state="error")
            st.error(f"Error ejecutando `{command}`:\n{e.stderr}")
            raise e


def initialize_st():
    """
    Inicializa TODO el pipeline RAG desde Streamlit.
    Ejecuta:
      1. ingest.py
      2. chunking.py
      3. build_index.py
      4. generate_eval_set.py
    """

    st.header("Inicializando pipeline BioActives RAG…")

    steps = [
        ("Ingesta de PDFs", "python -m src.ingest"),
        ("Generando chunks", "python -m src.chunking"),
        ("Construyendo índice vectorial", "python -m src.build_index"),
        ("Generando eval_set.json", "python -m src.generate_eval_set"),
    ]

    for title, cmd in steps:
        _run_step(title, cmd)

    st.success("Pipeline completado exitosamente.")
    st.balloons()
