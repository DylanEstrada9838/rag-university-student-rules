"""
dataset.py — Builds a RAGAS EvaluationDataset for the Reglamento Tec RAG pipeline.

Each sample contains:
  - user_input:          the question (from ground_truth.py)
  - reference:           manually written reference answer in Spanish
  - retrieved_contexts:  list of chunk strings (filled at runtime by running the retriever)
  - response:            the LLM-generated answer (filled at runtime by running the RAG chain)
"""

import os
import sys
import json

# Allow imports from the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ragas import EvaluationDataset
from ragas.dataset_schema import SingleTurnSample

# ── Manually written reference answers (Spanish, based on the document) ────────
# Each entry must match the question order in ground_truth.py exactly.
REFERENCE_ANSWERS = [
    # Q1 – Art. 43 – Baja definitiva
    (
        "La baja definitiva implica la separación permanente del estudiante de la institución "
        "y conlleva la pérdida de todos los derechos académicos. Esta medida se aplica como "
        "sanción disciplinaria en los casos de mayor gravedad previstos en el reglamento."
    ),
    # Q2 – Art. 36 – Consecuencias disciplinarias
    (
        "Las consecuencias disciplinarias que se pueden aplicar a un estudiante incluyen: "
        "amonestación, condicionamiento, suspensión temporal, baja temporal y baja definitiva. "
        "Estas medidas están ordenadas de menor a mayor gravedad según la falta cometida."
    ),
    # Q3 – Art. 40 / Art. 44b – ¿Quién impone el condicionamiento?
    (
        "El condicionamiento es una medida disciplinaria que limita la permanencia del "
        "estudiante en la institución bajo ciertas condiciones. Puede ser impuesto por el "
        "Director de Escuela o por el Comité Disciplinario, dependiendo de la gravedad "
        "de la falta cometida."
    ),
    # Q4 – Art. 50 – Medidas de protección
    (
        "Las medidas de protección son disposiciones provisionales adoptadas para resguardar "
        "la integridad, seguridad o bienestar de los miembros de la comunidad universitaria. "
        "Se aplican cuando existe un riesgo inminente o cuando es necesario preservar el "
        "orden y la convivencia durante el proceso disciplinario."
    ),
    # Q5 – Art. 53-54 – Comité Disciplinario
    (
        "El Comité Disciplinario se conforma por un representante del área académica, "
        "un representante de la dirección de vida estudiantil y, en algunos casos, "
        "un representante de la comunidad estudiantil. Su integración específica puede "
        "variar según el tipo de falta y está regulada en los artículos 53 y 54 del reglamento."
    ),
    # Q6 – Art. 35 – Arma dentro del campus
    (
        "Portar o introducir un arma dentro del campus universitario constituye una falta "
        "muy grave contemplada en el artículo 35 del reglamento. Esta conducta puede derivar "
        "en la baja definitiva del estudiante y en la notificación a las autoridades competentes."
    ),
    # Q7 – Art. 55 – Pruebas ante el Comité
    (
        "El estudiante puede presentar ante el Comité Disciplinario cualquier medio de prueba "
        "lícito, incluyendo documentos escritos, testimonios de testigos, evidencia digital, "
        "fotografías y cualquier otro elemento que considere pertinente para su defensa, "
        "conforme a lo establecido en el artículo 55 del reglamento."
    ),
    # Q8 – Art. 66-67 – Medidas de estabilización
    (
        "Las medidas de estabilización son acciones de apoyo orientadas a restablecer el "
        "equilibrio emocional, académico o social del estudiante. Se aplican cuando se "
        "detecta que el estudiante atraviesa una situación de vulnerabilidad o crisis que "
        "afecta su desempeño o convivencia, según lo dispuesto en los artículos 66 y 67."
    ),
    # Q9 – Art. 34.11 – Consumo de sustancias prohibidas
    (
        "Las faltas relacionadas con el consumo de sustancias prohibidas incluyen la "
        "introducción, posesión, distribución o consumo de alcohol, tabaco, drogas u otras "
        "sustancias controladas dentro de las instalaciones del campus. Estas conductas "
        "están tipificadas en el artículo 34.11 del reglamento y son consideradas faltas graves."
    ),
    # Q10 – Art. 36a / Art. 44a – Amonestación
    (
        "La amonestación es la sanción disciplinaria más leve y consiste en una llamada "
        "de atención formal al estudiante, quedando registrada en su expediente. "
        "Puede ser impuesta por el Director de Escuela o por el Director de Vida Estudiantil, "
        "según lo estipulado en los artículos 36a y 44a del reglamento."
    ),
]


def get_ground_truth_with_references():
    """
    Returns a list of dicts combining the questions from ground_truth.py
    with the manually written reference answers.
    """
    from testing.ground_truth import ground_truth  # noqa: E402

    assert len(ground_truth) == len(REFERENCE_ANSWERS), (
        f"Mismatch: {len(ground_truth)} questions vs {len(REFERENCE_ANSWERS)} references"
    )

    samples = []
    for item, ref in zip(ground_truth, REFERENCE_ANSWERS):
        samples.append({
            "question": item["question"],
            "expected_pages": item["expected_pages"],
            "reference": ref,
        })
    return samples


def build_evaluation_dataset(retriever, rag_chain):
    """
    Runs the RAG pipeline on all ground-truth questions and returns
    a RAGAS EvaluationDataset ready for evaluate().

    Args:
        retriever:  a LangChain retriever (already configured for the best config)
        rag_chain:  a LangChain chain that accepts a question string and returns an answer

    Returns:
        EvaluationDataset
    """
    cache_file = os.path.join(os.path.dirname(__file__), 'dataset_cache.json')
    if os.path.exists(cache_file):
        print(f"Loading cached dataset from {cache_file}...")
        with open(cache_file, 'r', encoding='utf-8') as f:
            cached_data = json.load(f)
        
        ragas_samples = [
            SingleTurnSample(
                user_input=item['user_input'],
                retrieved_contexts=item['retrieved_contexts'],
                response=item['response'],
                reference=item['reference']
            )
            for item in cached_data
        ]
        return EvaluationDataset(samples=ragas_samples)

    samples_meta = get_ground_truth_with_references()
    ragas_samples = []

    print("Building evaluation dataset…")
    for i, item in enumerate(samples_meta):
        question = item["question"]
        reference = item["reference"]

        # Retrieve contexts
        docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]

        # Generate answer with the RAG chain
        answer = rag_chain.invoke(question)

        ragas_samples.append(
            SingleTurnSample(
                user_input=question,
                retrieved_contexts=contexts,
                response=answer,
                reference=reference,
            )
        )
        print(f"  [{i + 1}/{len(samples_meta)}] '{question[:60]}…'")

    # Pre-caching dataset to prevent loss if the process is terminated
    cache_data = [
        {
            "user_input": s.user_input,
            "retrieved_contexts": s.retrieved_contexts,
            "response": s.response,
            "reference": s.reference,
        }
        for s in ragas_samples
    ]
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)

    print(f"Dataset ready and cached to {cache_file}\n")
    return EvaluationDataset(samples=ragas_samples)
