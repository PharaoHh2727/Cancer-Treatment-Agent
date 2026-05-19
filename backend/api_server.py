from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from urllib.parse import quote, unquote
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel


PROJECT_ROOT = Path(os.getenv("MEDICAL_AGENT_ROOT", Path(__file__).resolve().parents[1])).resolve()
load_dotenv(PROJECT_ROOT / ".env")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.general_agent import GeneralTool, run_general_assistant
import medical_tools

CLAM_DIR = Path(os.getenv("CLAM_DIR", PROJECT_ROOT / "CLAM-master")).resolve()
DATA_ROOT = Path(os.getenv("MEDICAL_AGENT_DATA_DIR", PROJECT_ROOT / "runtime_data")).resolve()
SESSIONS_DIR = DATA_ROOT / "sessions"
CSCO_DIR = Path(os.getenv("GUIDE_PDF_DIR", PROJECT_ROOT / "CSCO")).resolve()

AGENT_MODULE = None
AGENT_INSTANCE = None
AGENT_INSTANCES: dict[str, Any] = {}
AGENT_LOCK = threading.Lock()
ANALYSIS_TRACE_LOCK = threading.Lock()

LLM_MODEL_ALIASES = {
    "glm-4-flash": "glm-4-flash",
    "glm-4.7-flash": "glm-4.7-flash",
    "glm-z1-flash": "glm-z1-flash",
    "glm-4-plus": "glm-4-plus",
    "glm-4-air": "glm-4-air",
}


def log_api(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[API {timestamp}] {message}", flush=True)


class TeeStdout(io.TextIOBase):
    def __init__(self, stream: Any, buffer: io.StringIO):
        self.stream = stream
        self.buffer = buffer

    @property
    def encoding(self) -> str:
        return getattr(self.stream, "encoding", "utf-8")

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        self.stream.write(text)
        self.buffer.write(text)
        return len(text)

    def flush(self) -> None:
        self.stream.flush()


@contextlib.contextmanager
def capture_print_output():
    buffer = io.StringIO()
    with contextlib.redirect_stdout(TeeStdout(sys.stdout, buffer)):
        yield buffer


def trim_analysis_log(text: str) -> str:
    return (text or "").strip()


def literature_detail_score(text: str) -> int:
    raw = str(text or "")
    if not raw.strip():
        return 0
    score = 0
    if "PMID:" in raw:
        score += 1
    if "【PubMed-" in raw or "PubMed-" in raw:
        score += 2
    detail_markers = [
        "题名:",
        "作者:",
        "期刊:",
        "研究类型:",
        "治疗相关证据摘要:",
        "引用标注:",
        "DOI:",
        "链接:",
    ]
    score += sum(1 for marker in detail_markers if marker in raw)
    return score


def choose_more_detailed_literature(*candidates: Any) -> str:
    best = ""
    best_score = 0
    for candidate in candidates:
        text = str(candidate or "").strip()
        score = literature_detail_score(text)
        if score > best_score or (score == best_score and len(text) > len(best)):
            best = text
            best_score = score
    return best


def extract_pubmed_evidence_section(text: str) -> str:
    raw = str(text or "")
    match = re.search(
        r"【PubMed文献证据】\s*([\s\S]*?)(?=\n\s*【治疗方案质检警告】|\n\s*【最终治疗方案】|\Z)",
        raw,
    )
    return match.group(1).strip() if match else ""


def parse_cors_origins() -> list[str]:
    raw = os.getenv(
        "MEDICAL_AGENT_CORS_ORIGINS",
        "http://127.0.0.1:5173,http://localhost:5173,http://0.0.0.0:5173",
    )
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    return origins or ["*"]


app = FastAPI(title="Medical Agent API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ClamRunRequest(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    session_id: str
    query: str
    cancer_type: str | None = None
    llm_model: str | None = None


def safe_filename(filename: str) -> str:
    name = Path(filename or "upload").name
    name = re.sub(r"[^A-Za-z0-9._\-\u4e00-\u9fff]+", "_", name).strip("._")
    return name or f"upload_{uuid.uuid4().hex}"


def new_session_id() -> str:
    return uuid.uuid4().hex


def session_dir(session_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-fA-F]{32}", session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id")
    path = SESSIONS_DIR / session_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def state_path(session_id: str) -> Path:
    return session_dir(session_id) / "state.json"


def default_state(session_id: str) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "cancer_type": "BRCA",
        "llm_model": normalize_llm_model(None),
        "patient_summary": {},
        "uploaded_files": {},
        "created_at": time.time(),
        "updated_at": time.time(),
    }


def load_state(session_id: str) -> dict[str, Any]:
    path = state_path(session_id)
    if not path.exists():
        return default_state(session_id)
    with path.open("r", encoding="utf-8") as f:
        state = json.load(f)
    state.setdefault("session_id", session_id)
    state.setdefault("cancer_type", "BRCA")
    state.setdefault("llm_model", normalize_llm_model(None))
    state.setdefault("patient_summary", {})
    state.setdefault("uploaded_files", {})
    return state


def save_state(state: dict[str, Any]) -> None:
    state["updated_at"] = time.time()
    path = state_path(state["session_id"])
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def save_upload(upload: UploadFile, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as f:
        shutil.copyfileobj(upload.file, f)


def infer_cancer_type_from_csv(csv_path: Path, default: str = "BRCA") -> str:
    try:
        df = pd.read_csv(csv_path, nrows=1)
        if "oncotree_code" not in df.columns or df.empty:
            return default
        value = str(df.loc[0, "oncotree_code"]).strip().upper()
        return value or default
    except Exception:
        return default


def read_patient_summary(csv_path: Path) -> dict[str, str]:
    fields = ["case_id", "s_female", "is_female", "oncotree_code", "age"]
    labels = {
        "case_id": "Case ID",
        "s_female": "Sex",
        "is_female": "Sex",
        "oncotree_code": "Cancer Type",
        "age": "Age",
    }
    try:
        df = pd.read_csv(csv_path, nrows=1)
    except Exception:
        return {}

    if df.empty:
        return {}

    summary: dict[str, str] = {}
    row = df.iloc[0]
    for field in fields:
        if field not in df.columns:
            continue
        value = row[field]
        if pd.isna(value):
            continue
        if field in {"s_female", "is_female"}:
            value = "female" if int(float(value)) == 1 else "male"
        elif field == "age":
            value = str(int(value)) if float(value).is_integer() else str(value)
        else:
            value = str(value)
        summary[labels[field]] = value
    return summary


def run_command(command: list[str], cwd: Path | None = None) -> str:
    log_api("Running: " + " ".join(command))
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=False,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stdout or f"Command failed: {command[0]}")
    return result.stdout


def run_clam_pipeline(state: dict[str, Any]) -> dict[str, Any]:
    files = state.get("uploaded_files", {})
    slide_path = Path(files.get("pathology_wsi", ""))
    if not slide_path.exists():
        raise ValueError("请先上传病理 WSI 文件。")
    if not CLAM_DIR.exists():
        raise FileNotFoundError(f"CLAM directory not found: {CLAM_DIR}")

    sid = state["session_id"]
    base = session_dir(sid)
    source_dir = base / "clam_source"
    patch_output_dir = base / "clam_output"
    feat_output_dir = base / "clam_features"
    source_dir.mkdir(exist_ok=True)
    patch_output_dir.mkdir(exist_ok=True)
    feat_output_dir.mkdir(exist_ok=True)

    slide_copy = source_dir / slide_path.name
    if slide_copy.resolve() != slide_path.resolve():
        shutil.copy2(slide_path, slide_copy)

    clam_python = os.getenv("CLAM_PYTHON", sys.executable)
    slide_ext = slide_copy.suffix or ".svs"
    patch_log = run_command(
        [
            clam_python,
            str(CLAM_DIR / "create_patches_fp.py"),
            "--source",
            str(source_dir),
            "--save_dir",
            str(patch_output_dir),
            "--patch_size",
            os.getenv("CLAM_PATCH_SIZE", "256"),
            "--seg",
            "--patch",
            "--stitch",
        ],
        cwd=CLAM_DIR,
    )

    slides_csv = base / "slides.csv"
    pd.DataFrame({"slide_id": [slide_copy.stem]}).to_csv(slides_csv, index=False)

    feature_log = run_command(
        [
            clam_python,
            str(CLAM_DIR / "extract_features_fp.py"),
            "--data_h5_dir",
            str(patch_output_dir),
            "--data_slide_dir",
            str(source_dir),
            "--csv_path",
            str(slides_csv),
            "--feat_dir",
            str(feat_output_dir),
            "--batch_size",
            os.getenv("CLAM_BATCH_SIZE", "512"),
            "--slide_ext",
            slide_ext,
        ],
        cwd=CLAM_DIR,
    )

    mask_files = sorted((patch_output_dir / "masks").glob("*.jpg"))
    pt_files = sorted((feat_output_dir / "pt_files").glob("*.pt"))
    if not mask_files:
        raise FileNotFoundError(f"CLAM mask jpg not found under {patch_output_dir / 'masks'}")
    if not pt_files:
        raise FileNotFoundError(f"CLAM pt feature file not found under {feat_output_dir / 'pt_files'}")

    mask_jpg = mask_files[0]
    pt_file = pt_files[0]
    files["pathology_jpg"] = str(mask_jpg)
    files["pathology_pt"] = str(pt_file)
    state["uploaded_files"] = files
    save_state(state)

    return {
        "session_id": sid,
        "pathology_jpg": str(mask_jpg),
        "pathology_jpg_url": make_file_url(sid, mask_jpg),
        "pathology_pt": str(pt_file),
        "logs": {
            "create_patches_fp": patch_log[-4000:],
            "extract_features_fp": feature_log[-4000:],
        },
    }


def load_agent_module():
    global AGENT_MODULE
    if AGENT_MODULE is None:
        AGENT_MODULE = importlib.import_module("autonomous_agent")
    return AGENT_MODULE


def normalize_llm_model(model: str | None) -> str:
    default_model = os.getenv("DEFAULT_LLM_MODEL", "glm-4-flash")
    raw = (model or default_model).strip()
    return LLM_MODEL_ALIASES.get(raw.lower(), LLM_MODEL_ALIASES.get(default_model.lower(), "glm-4-flash"))


def get_agent(llm_model: str | None = None):
    global AGENT_INSTANCE
    model_name = normalize_llm_model(llm_model)
    with AGENT_LOCK:
        module = load_agent_module()
        if hasattr(module, "set_active_llm_model"):
            module.set_active_llm_model(model_name)
        if model_name not in AGENT_INSTANCES:
            AGENT_INSTANCES[model_name] = module.create_autonomous_agent(verbose=True, llm_model=model_name)
        AGENT_INSTANCE = AGENT_INSTANCES[model_name]
        return AGENT_INSTANCE


def build_agent_query(query: str, state: dict[str, Any]) -> str:
    files = state.get("uploaded_files", {})
    full_query = query.strip()
    if files.get("pathology_pt"):
        full_query += f" 病理特征文件: {files['pathology_pt']}"
    if files.get("genome"):
        full_query += f" 基因组数据文件: {files['genome']}"
    return full_query


def build_general_runtime_context(
    state: dict[str, Any],
    cancer_type: str,
    medical_llm_model: str,
) -> dict[str, Any]:
    files = state.get("uploaded_files", {})
    return {
        "session_id": state.get("session_id"),
        "cancer_type": cancer_type,
        "medical_llm_model": medical_llm_model,
        "patient_summary": state.get("patient_summary", {}),
        "available_patient_data": {
            "pathology_wsi_uploaded": bool(files.get("pathology_wsi")),
            "pathology_pt_ready": bool(files.get("pathology_pt")),
            "genome_csv_uploaded": bool(files.get("genome")),
        },
        "system_tools": [
            "daily_qa_direct_answer",
            "predict_tnm_stage",
            "predict_survival",
            "medical_treatment_plan",
            "CSCO_RAG",
            "PubMed_search",
            "CLAM_pathology_feature_pipeline",
        ],
    }


def build_general_tools(
    query: str,
    state: dict[str, Any],
    cancer_type: str,
    llm_model: str,
    tool_state: dict[str, Any],
) -> list[GeneralTool]:
    files = state.get("uploaded_files", {})

    def load_medical_agent_module():
        module = load_agent_module()
        if hasattr(module, "set_active_llm_model"):
            module.set_active_llm_model(llm_model)
        return module

    def selected_cancer_type(args: dict[str, Any]) -> str:
        return str(args.get("cancer_type") or cancer_type or "BRCA").upper()

    def run_medical_react(args: dict[str, Any], fallback_request: str, task_goal: str) -> str:
        request_text = str(args.get("request") or fallback_request).strip()
        load_medical_agent_module()
        agent = get_agent(llm_model)
        result = agent.run(
            build_agent_query(request_text, state),
            selected_cancer_type(args),
            task_goal=task_goal,
        )
        literature_results = str(getattr(agent, "last_literature_results", "") or "")
        agent_state = {}
        if hasattr(agent, "build_evaluation_record"):
            try:
                agent_state = agent.build_evaluation_record(name=f"session_{state.get('session_id')}")
            except Exception:
                agent_state = {}
        treatment_context = medical_tools.LAST_TREATMENT_CONTEXT if task_goal == "treatment" else {}
        literature_results = choose_more_detailed_literature(
            literature_results,
            agent_state.get("literature_results") if isinstance(agent_state, dict) else "",
            treatment_context.get("literature_results"),
            extract_pubmed_evidence_section(str(result)),
        )
        if literature_results:
            tool_state["literature_results"] = literature_results
        if agent_state:
            tool_state["medical_agent_state"] = agent_state
        return str(result)

    def predict_stage(args: dict[str, Any]) -> str:
        return run_medical_react(args, query, "stage")

    def predict_survival(args: dict[str, Any]) -> str:
        return run_medical_react(args, query, "survival")

    def medical_treatment_plan(args: dict[str, Any]) -> str:
        return run_medical_react(args, query, "treatment")

    return [
        GeneralTool(
            name="predict_tnm_stage",
            description=(
                "进入医学 ReAct Agent 执行当前病例TNM/临床分期预测。"
                "医学 Agent 会在ReAct循环中选择并调用本地 TransMIL 分期工具。"
            ),
            input_schema={
                "request": "可选；用户分期预测需求原文或精炼描述",
                "cancer_type": "BRCA | BLCA | LUAD，可选；默认使用当前病例癌种",
            },
            handler=predict_stage,
        ),
        GeneralTool(
            name="predict_survival",
            description=(
                "进入医学 ReAct Agent 执行当前病例生存时间/风险预测。"
                "医学 Agent 会在ReAct循环中选择并调用本地 CMTA 生存预测工具。"
            ),
            input_schema={
                "request": "可选；用户生存预测需求原文或精炼描述",
                "cancer_type": "BRCA | BLCA | LUAD，可选；默认使用当前病例癌种",
            },
            handler=predict_survival,
        ),
        GeneralTool(
            name="medical_treatment_plan",
            description=(
                "进入医学 ReAct Agent 生成当前病例个体化治疗方案。该智能体内部会按需调用分期、生存、PubMed 检索、"
                "CSCO RAG 和治疗方案生成工具。仅用于用户明确需要当前病例的个体化治疗方案时。"
            ),
            input_schema={
                "request": "可选；对用户治疗方案需求的原文或精炼描述",
                "cancer_type": "BRCA | BLCA | LUAD，可选；默认使用当前病例癌种",
            },
            handler=medical_treatment_plan,
        ),
    ]


def make_file_url(session_id: str, file_path: Path) -> str:
    base = session_dir(session_id).resolve()
    rel = file_path.resolve().relative_to(base).as_posix()
    return f"/api/files/{session_id}/{rel}"


def safe_csco_pdf_path(filename: str) -> Path:
    decoded = unquote(filename)
    if "/" in decoded or "\\" in decoded:
        raise HTTPException(status_code=400, detail="Invalid file name")
    target = (CSCO_DIR / decoded).resolve()
    if CSCO_DIR not in [target, *target.parents]:
        raise HTTPException(status_code=403, detail="Forbidden path")
    if not target.is_file() or target.suffix.lower() != ".pdf":
        raise HTTPException(status_code=404, detail="PDF file not found")
    return target


def csco_file_meta(path: Path) -> dict[str, Any]:
    return {
        "name": path.name,
        "size": path.stat().st_size,
        "url": f"/api/knowledge/csco/{quote(path.name)}",
    }


def list_csco_file_metas() -> list[dict[str, Any]]:
    if not CSCO_DIR.exists():
        return []
    return [csco_file_meta(path) for path in sorted(CSCO_DIR.glob("*.pdf"))]


def clear_csco_page_cache(pdf_path: Path) -> None:
    cache_dir = DATA_ROOT / "csco_page_cache"
    if not cache_dir.exists():
        return
    cache_prefix = f"{safe_filename(pdf_path.stem)}_page_"
    for path in cache_dir.glob(f"{cache_prefix}*.png"):
        try:
            path.unlink()
        except OSError:
            pass


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "project_root": str(PROJECT_ROOT),
        "data_root": str(DATA_ROOT),
        "clam_dir": str(CLAM_DIR),
        "clam_exists": CLAM_DIR.exists(),
        "csco_dir": str(CSCO_DIR),
        "csco_exists": CSCO_DIR.exists(),
        "zhipuai_api_key_set": bool(os.getenv("ZHIPUAI_API_KEY")),
        "pubmed_api_key_set": bool(os.getenv("PUBMED_API_KEY")),
    }


@app.get("/api/knowledge/csco")
def list_csco_pdfs() -> dict[str, Any]:
    return {"files": list_csco_file_metas()}


@app.post("/api/knowledge/csco")
def upload_csco_pdf(file: UploadFile = File(...)) -> dict[str, Any]:
    filename = safe_filename(file.filename or "knowledge.pdf")
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="CSCO 知识库目前仅支持 PDF 文件")

    CSCO_DIR.mkdir(parents=True, exist_ok=True)
    target = (CSCO_DIR / filename).resolve()
    if CSCO_DIR not in [target, *target.parents]:
        raise HTTPException(status_code=403, detail="Forbidden path")

    save_upload(file, target)
    clear_csco_page_cache(target)
    log_api(f"CSCO knowledge PDF uploaded: {target}")
    return {"file": csco_file_meta(target), "files": list_csco_file_metas()}


@app.get("/api/knowledge/csco/{filename}")
def get_csco_pdf(filename: str):
    target = safe_csco_pdf_path(filename)
    response = FileResponse(target, media_type="application/pdf")
    response.headers["Content-Disposition"] = f"inline; filename*=UTF-8''{quote(target.name)}"
    return response


@app.delete("/api/knowledge/csco/{filename}")
def delete_csco_pdf(filename: str) -> dict[str, Any]:
    target = safe_csco_pdf_path(filename)
    deleted_name = target.name
    try:
        target.unlink()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"删除 PDF 文件失败: {exc}") from exc

    clear_csco_page_cache(target)
    log_api(f"CSCO knowledge PDF deleted: {target}")
    return {"deleted": deleted_name, "files": list_csco_file_metas()}


@app.get("/api/knowledge/csco/{filename}/pages/{page_number}")
def get_csco_pdf_page_image(filename: str, page_number: int):
    target = safe_csco_pdf_path(filename)
    if page_number < 1:
        raise HTTPException(status_code=400, detail="Invalid page number")

    try:
        import fitz  # type: ignore
    except Exception as exc:
        raise HTTPException(status_code=501, detail="PDF page rendering requires PyMuPDF") from exc

    cache_dir = DATA_ROOT / "csco_page_cache"
    cache_name = f"{safe_filename(target.stem)}_page_{page_number}.png"
    cache_path = cache_dir / cache_name
    if cache_path.exists() and cache_path.stat().st_mtime >= target.stat().st_mtime:
        return FileResponse(cache_path, media_type="image/png")

    try:
        with fitz.open(target) as doc:
            if page_number > doc.page_count:
                raise HTTPException(status_code=404, detail="PDF page not found")
            page = doc.load_page(page_number - 1)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False)
            cache_dir.mkdir(parents=True, exist_ok=True)
            pixmap.save(cache_path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to render PDF page: {exc}") from exc

    return FileResponse(cache_path, media_type="image/png")


@app.get("/api/session/{session_id}/literature")
def get_session_literature(session_id: str) -> dict[str, Any]:
    state = load_state(session_id)
    agent_state = state.get("medical_agent_state") if isinstance(state.get("medical_agent_state"), dict) else {}
    literature = choose_more_detailed_literature(
        state.get("literature_results"),
        state.get("literature_results_full"),
        agent_state.get("literature_results") if isinstance(agent_state, dict) else "",
    )
    return {
        "session_id": session_id,
        "has_literature": bool(literature.strip()),
        "literature": literature,
    }


@app.post("/api/session")
def create_session() -> dict[str, str]:
    sid = new_session_id()
    save_state(default_state(sid))
    return {"session_id": sid}


@app.post("/api/upload/pathology")
def upload_pathology(
    file: UploadFile = File(...),
    session_id: str | None = Form(default=None),
) -> dict[str, Any]:
    sid = session_id or new_session_id()
    state = load_state(sid)
    filename = safe_filename(file.filename or "pathology.svs")
    target = session_dir(sid) / "uploads" / filename
    save_upload(file, target)

    files = state.setdefault("uploaded_files", {})
    files["pathology_wsi"] = str(target)
    files.pop("pathology_jpg", None)
    files.pop("pathology_pt", None)
    save_state(state)
    log_api(f"Pathology uploaded: session={sid}, path={target}")
    return {"session_id": sid, "file_name": filename, "file_path": str(target)}


@app.post("/api/upload/genome")
def upload_genome(
    file: UploadFile = File(...),
    session_id: str | None = Form(default=None),
) -> dict[str, Any]:
    sid = session_id or new_session_id()
    state = load_state(sid)
    filename = safe_filename(file.filename or "genome.csv")
    target = session_dir(sid) / "uploads" / filename
    save_upload(file, target)

    cancer_type = infer_cancer_type_from_csv(target, state.get("cancer_type", "BRCA"))
    patient_summary = read_patient_summary(target)
    files = state.setdefault("uploaded_files", {})
    files["genome"] = str(target)
    state["cancer_type"] = cancer_type
    state["patient_summary"] = patient_summary
    save_state(state)
    log_api(f"Genome uploaded: session={sid}, cancer_type={cancer_type}, path={target}")
    return {
        "session_id": sid,
        "file_name": filename,
        "file_path": str(target),
        "cancer_type": cancer_type,
        "patient_summary": patient_summary,
    }


@app.post("/api/clam/run")
def run_clam(request: ClamRunRequest) -> dict[str, Any]:
    state = load_state(request.session_id)
    try:
        return run_clam_pipeline(state)
    except Exception as exc:
        log_api(f"CLAM failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/agent/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    state = load_state(request.session_id)
    cancer_type = (request.cancer_type or state.get("cancer_type") or "BRCA").upper()
    llm_model = normalize_llm_model(request.llm_model or state.get("llm_model"))
    state["cancer_type"] = cancer_type
    state["llm_model"] = llm_model
    save_state(state)

    files = state.get("uploaded_files", {})
    tool_state: dict[str, Any] = {}
    runtime_context = build_general_runtime_context(state, cancer_type, llm_model)
    general_tools = build_general_tools(query, state, cancer_type, llm_model, tool_state)
    log_api(
        f"Chat request: session={request.session_id}, route=general_supervisor, "
        f"cancer_type={cancer_type}, llm_model={llm_model}, has_pt={bool(files.get('pathology_pt'))}, "
        f"has_genome={bool(files.get('genome'))}"
    )

    literature_results = ""
    analysis_log = ""
    supervisor_result: dict[str, Any] = {}
    try:
        with ANALYSIS_TRACE_LOCK, capture_print_output() as trace:
            supervisor_result = run_general_assistant(
                query=query,
                tools=general_tools,
                runtime_context=runtime_context,
                llm_model=llm_model,
            )
            result = supervisor_result.get("answer") or "主智能体没有返回有效回答。"
            literature_results = str(tool_state.get("literature_results") or "")
            analysis_log = trim_analysis_log(trace.getvalue())
    except Exception as exc:
        if "trace" in locals():
            analysis_log = trim_analysis_log(trace.getvalue())
        log_api(f"Agent failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    medical_agent_literature = (
        tool_state.get("medical_agent_state", {}).get("literature_results")
        if isinstance(tool_state.get("medical_agent_state"), dict)
        else ""
    )
    literature_for_response = choose_more_detailed_literature(
        literature_results,
        extract_pubmed_evidence_section(str(result)),
        medical_agent_literature,
    )
    if literature_for_response:
        state = load_state(request.session_id)
        state["literature_results"] = choose_more_detailed_literature(
            state.get("literature_results"),
            literature_for_response,
        )
        state["literature_results_full"] = state["literature_results"]
        if isinstance(tool_state.get("medical_agent_state"), dict):
            state["medical_agent_state"] = tool_state["medical_agent_state"]
        save_state(state)
        literature_for_response = state["literature_results"]

    return {
        "session_id": request.session_id,
        "intent": str(supervisor_result.get("route") or "general"),
        "cancer_type": cancer_type,
        "llm_model": llm_model,
        "general_model": supervisor_result.get("general_model"),
        "tool_calls": supervisor_result.get("tool_calls", []),
        "result": str(result),
        "literature": literature_for_response,
        "analysis_log": analysis_log,
    }


@app.get("/api/files/{session_id}/{relative_path:path}")
def get_file(session_id: str, relative_path: str):
    base = session_dir(session_id).resolve()
    target = (base / relative_path).resolve()
    if base not in [target, *target.parents]:
        raise HTTPException(status_code=403, detail="Forbidden path")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("MEDICAL_AGENT_HOST", "0.0.0.0")
    port = int(os.getenv("MEDICAL_AGENT_API_PORT", os.getenv("PORT", "8000")))
    uvicorn.run("backend.api_server:app", host=host, port=port, reload=False)
