# Ground truth question-answer pairs for evaluating the retriever.
# Each entry contains a question and the expected metadata page(s) where the answer is found.
# NOTE: Page numbers correspond to doc.metadata["page"] from PyMuPDFLoader (0-indexed),
#       NOT the printed page number in the PDF.

ground_truth = [
    {
        "question": "¿Qué implica la baja definitiva de un estudiante?",
        "expected_pages": [47],  # Art. 43 – metadata page 47
    },
    {
        "question": "¿Cuáles son las consecuencias disciplinarias que se pueden aplicar a un estudiante?",
        "expected_pages": [46],  # Art. 36 – metadata page 46 (classification list)
    },
    {
        "question": "¿Quién puede imponer un condicionamiento a un estudiante?",
        "expected_pages": [47, 50],  # Art. 40 (pg 47) defines it; Art. 44b (pg 50) who imposes
    },
    {
        "question": "¿Qué son las medidas de protección y cuándo se aplican?",
        "expected_pages": [53],  # Art. 50 – metadata page 53
    },
    {
        "question": "¿Cómo se conforma un Comité Disciplinario?",
        "expected_pages": [54, 55],  # Art. 53-54 – metadata pages 54-55
    },
    {
        "question": "¿Qué pasa si un estudiante es sorprendido con un arma dentro del campus?",
        "expected_pages": [43],  # Art. 35 – metadata page 43
    },
    {
        "question": "¿Qué pruebas puede presentar un estudiante ante el Comité Disciplinario?",
        "expected_pages": [56],  # Art. 55 – metadata page 56
    },
    {
        "question": "¿Qué son las medidas de estabilización y cuándo se aplican?",
        "expected_pages": [63, 64],  # Art. 66 (pg 63) intro; Art. 67 (pg 64) details
    },
    {
        "question": "¿Cuáles son las faltas relacionadas con el consumo de sustancias prohibidas?",
        "expected_pages": [41],  # Art. 34.11 – metadata page 41
    },
    {
        "question": "¿Qué es una amonestación y quién puede imponerla?",
        "expected_pages": [46, 50],  # Art. 36a (pg 46) defines; Art. 44a (pg 50) who imposes
    },
]
