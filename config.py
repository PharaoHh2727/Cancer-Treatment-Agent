import os
from pathlib import Path


# Project paths. Defaults are relative to the repository root so that the
# project can run on another user's machine after cloning the repository.
BASE_DIR = Path(os.getenv("MEDICAL_AGENT_ROOT", Path(__file__).parent)).resolve()
MODEL_ROOT = Path(os.getenv("MEDICAL_AGENT_MODEL_DIR", BASE_DIR)).resolve()
GENE_DATA_DIR = Path(os.getenv("GENE_DATA_DIR", BASE_DIR / "gene_data")).resolve()

# TransMIL TNM model checkpoints.
TRANSMIL_MODEL_DIR = Path(
    os.getenv("TRANSMIL_MODEL_DIR", MODEL_ROOT / "transmil_tnm" / "results")
).resolve()
TRANSMIL_MODEL_PATH = {
    "BLCA": {
        "result_dir": os.path.join(TRANSMIL_MODEL_DIR, "BLCA/epoch=17-val_loss=0.3275.ckpt"),
    },
    "BRCA": {
        "result_dir": os.path.join(TRANSMIL_MODEL_DIR, "BRCA/epoch=11-val_loss=0.3178.ckpt"),
    },
    "LUAD": {
        "result_dir": os.path.join(TRANSMIL_MODEL_DIR, "LUAD/epoch=01-val_loss=0.3125.ckpt"),
    },
}

# CMTA survival model checkpoints.
CMTA_RESULTS_DIR = str(
    Path(os.getenv("CMTA_RESULTS_DIR", MODEL_ROOT / "cmta_survival" / "results")).resolve()
)
MODEL_CONFIGS = {
    "BLCA": {
        "result_dir": os.path.join(CMTA_RESULTS_DIR, "BLCA"),
    },
    "BRCA": {
        "result_dir": os.path.join(CMTA_RESULTS_DIR, "BRCA"),
    },
    "LUAD": {
        "result_dir": os.path.join(CMTA_RESULTS_DIR, "LUAD"),
    },
}

# Reference genomic data used by the CMTA preprocessing code.
SIGNATURES_CSV_PATH = Path(os.getenv("SIGNATURES_CSV_PATH", GENE_DATA_DIR / "signatures.csv")).resolve()

# RAG knowledge base and local CSCO guideline PDFs.
RAG_DB_DIR = Path(os.getenv("RAG_DB_DIR", BASE_DIR / "chroma_db")).resolve()
GUIDE_PDF_DIR = Path(os.getenv("GUIDE_PDF_DIR", BASE_DIR / "CSCO")).resolve()
CANCER_TYPES = {
    "BRCA": "2025CSCO乳腺癌诊疗指南_OCR.pdf",
    "LUAD": "2025CSCO非小细胞肺癌诊疗指南_OCR.pdf",
    "BLCA": "2025CSCO尿路上皮癌诊疗指南_OCR.pdf",
}

# TNM classification settings.
T_CLASSES = 4
N_CLASSES = 4
M_CLASSES = 2
STAGE_CLASSES = 4
T_LABELS = ["T1", "T2", "T3", "T4"]
N_LABELS = ["N0", "N1", "N2", "N3"]
M_LABELS = ["M0", "M1"]
STAGE_LABELS = ["Stage I", "Stage II", "Stage III", "Stage IV"]

# RAG and LLM settings.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "6"))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
RAG_USE_VECTOR_DB = os.getenv("RAG_USE_VECTOR_DB", "true").lower() not in {"0", "false", "no"}
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "GLM-4-Flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.5"))
VERBOSE_REASONING = os.getenv("VERBOSE_REASONING", "true").lower() not in {"0", "false", "no"}
MAX_REASONING_STEPS = int(os.getenv("MAX_REASONING_STEPS", "10"))

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "results")).resolve()
