"""
医学工具模块。
提供分期预测、生存预测、PubMed检索和治疗方案生成工具。
"""
import os
import re
import json
import time
from contextvars import ContextVar
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

# 设置HF镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from dotenv import load_dotenv
load_dotenv()

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

import config
import torch
import numpy as np
from pubmed.pubmed_search import search_cancer_literature, format_pubmed_results
from transmil_tnm import get_transmil_predictor
from cmta_survival import get_predictor
from rag import MedicalRAG

LAST_TREATMENT_CONTEXT: Dict[str, Any] = {}
_ACTIVE_PUBMED_EVIDENCE: ContextVar[str] = ContextVar("active_pubmed_evidence", default="")

LLM_RETRY_MAX_ATTEMPTS = int(os.getenv("ZHIPUAI_MAX_RETRIES", "4"))
LLM_RETRY_BASE_SECONDS = float(os.getenv("ZHIPUAI_RETRY_BASE_SECONDS", "3"))
LLM_RETRY_MAX_SECONDS = float(os.getenv("ZHIPUAI_RETRY_MAX_SECONDS", "30"))
LLM_ERROR_BODY_MAX_CHARS = int(os.getenv("ZHIPUAI_ERROR_BODY_MAX_CHARS", "4000"))
ACTIVE_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", config.DEFAULT_LLM_MODEL)
TREATMENT_PLAN_DEBUG = os.getenv("TREATMENT_PLAN_DEBUG", "1").lower() not in {"0", "false", "no", "off"}
TREATMENT_PLAN_DEBUG_CHARS = int(os.getenv("TREATMENT_PLAN_DEBUG_CHARS", "6000"))


def set_active_llm_model(model_name: str):
    """Set the runtime LLM model used by agent-created LLM calls."""
    global ACTIVE_LLM_MODEL
    if model_name:
        ACTIVE_LLM_MODEL = model_name

def get_active_llm_model() -> str:
    return ACTIVE_LLM_MODEL or config.DEFAULT_LLM_MODEL

CANCER_NAME_CN = {
    "BRCA": "乳腺癌",
    "BLCA": "尿路上皮癌 膀胱癌",
    "LUAD": "非小细胞肺癌 肺腺癌",
}

CSCO_RAG_QUERY_BY_CANCER = {
    "BLCA": "膀胱癌 肌层浸润性膀胱癌 局部晚期 根治性膀胱切除 新辅助治疗 辅助治疗 术后辅助治疗 一线治疗 二线治疗",
    "BRCA": "乳腺癌 分期 分层 手术 新辅助治疗 辅助治疗 内分泌治疗 化疗 抗HER2治疗 放疗 复发转移治疗",
    "LUAD": "非小细胞肺癌 肺腺癌 分期 驱动基因检测 手术 新辅助治疗 辅助治疗 靶向治疗 免疫治疗 化疗 放疗 一线治疗 后线治疗",
}

#检查LLM报错原因
def _get_llm_model_name(llm) -> str:
    for attr in ("model", "model_name", "model_name_"):
        value = getattr(llm, attr, None)
        if value:
            return str(value)
    return llm.__class__.__name__

def _message_content_for_debug(message: Any) -> str:
    if hasattr(message, "content"):
        return str(message.content or "")
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(message)

def _payload_debug_size(payload: Any) -> int:
    if isinstance(payload, str):
        return len(payload)
    if isinstance(payload, list):
        return sum(len(_message_content_for_debug(item)) for item in payload)
    return len(str(payload))

def _safe_response_body(response) -> str:
    body = getattr(response, "text", "") or ""
    try:
        parsed = json.loads(body)
        body = json.dumps(parsed, ensure_ascii=False, indent=2)
    except Exception:
        pass
    if len(body) > LLM_ERROR_BODY_MAX_CHARS:
        return body[:LLM_ERROR_BODY_MAX_CHARS] + "\n...（响应正文过长，已截断）"
    return body

def _filtered_response_headers(response) -> Dict[str, str]:
    if response is None:
        return {}
    interesting = {
        "date",
        "content-type",
        "content-length",
        "retry-after",
        "x-request-id",
        "x-ratelimit-limit-requests",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-reset-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-tokens",
    }
    headers = getattr(response, "headers", {}) or {}
    return {
        key: value
        for key, value in headers.items()
        if key.lower() in interesting
    }

def _print_llm_error_diagnostics(llm, payload, call_name: str, exc: Exception, attempt: int):
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    print("\n" + "=" * 60, flush=True)
    print(f"[LLM调用诊断] {call_name} 第{attempt + 1}次失败", flush=True)
    print(f"- model: {_get_llm_model_name(llm)}", flush=True)
    print(f"- payload_chars: {_payload_debug_size(payload)}", flush=True)
    print(f"- exception_type: {exc.__class__.__name__}", flush=True)
    print(f"- exception: {exc}", flush=True)
    if status_code is not None:
        print(f"- http_status: {status_code}", flush=True)
    headers = _filtered_response_headers(response)
    if headers:
        print(f"- response_headers: {json.dumps(headers, ensure_ascii=False)}", flush=True)
    if response is not None:
        body = _safe_response_body(response)
        print("- response_body:", flush=True)
        print(body if body else "  <empty>", flush=True)
    print("=" * 60 + "\n", flush=True)

def _is_retryable_llm_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code == 429 or (status_code is not None and 500 <= status_code < 600):
        return True

    message = str(exc).lower()
    retry_markers = [
        "429",
        "too many requests",
        "rate limit",
        "read operation timed out",
        "connect timeout",
        "temporarily unavailable",
    ]
    return any(marker in message for marker in retry_markers)

def _invoke_llm_with_retry(llm, payload, call_name: str = "LLM"):
    for attempt in range(LLM_RETRY_MAX_ATTEMPTS + 1):
        try:
            return llm.invoke(payload)
        except Exception as exc:
            _print_llm_error_diagnostics(llm, payload, call_name, exc, attempt)
            if attempt >= LLM_RETRY_MAX_ATTEMPTS or not _is_retryable_llm_error(exc):
                raise

            wait_seconds = min(
                LLM_RETRY_BASE_SECONDS * (2 ** attempt),
                LLM_RETRY_MAX_SECONDS,
            )
            print(
                f"[{call_name}] 第{attempt + 1}次调用失败，{wait_seconds:.1f}s后重试: {exc}",
                flush=True,
            )
            time.sleep(wait_seconds)


#检查生成的治疗方案是否满足基本的格式和内容要求，返回缺失项列表
def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return "\n".join(_to_text(item) for item in value if item is not None)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)

def set_active_pubmed_evidence(evidence: Any):
    """Set full PubMed evidence for the current treatment-plan tool call."""
    return _ACTIVE_PUBMED_EVIDENCE.set(_to_text(evidence).strip())

def get_active_pubmed_evidence() -> str:
    return _to_text(_ACTIVE_PUBMED_EVIDENCE.get()).strip()

def reset_active_pubmed_evidence(token=None) -> None:
    if token is None:
        _ACTIVE_PUBMED_EVIDENCE.set("")
        return
    _ACTIVE_PUBMED_EVIDENCE.reset(token)

def _remove_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", _to_text(text), flags=re.S | re.I).strip()

def _strip_think_tags(text: str) -> str:
    return re.sub(r"</?think>", "", _to_text(text), flags=re.I).strip()

def _normalize_stage(value: Any) -> str:
    text = _to_text(value).strip()
    if not text:
        return ""
    return text if text.lower().startswith("stage") else f"Stage {text}"

def _normalize_tnm(value: Any, prefix: str) -> str:
    text = _to_text(value).strip()
    if not text:
        return ""
    return text if text.upper().startswith(prefix.upper()) else f"{prefix}{text}"

def _extract_pmids(text: str) -> List[str]:
    return sorted(set(re.findall(r"PMID:\s*(\d+)", _to_text(text))))

def _coerce_pmids(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        values = []
        for item in value:
            item_text = _to_text(item)
            values.extend(re.findall(r"\d{5,}", item_text))
    else:
        values = re.findall(r"\d{5,}", _to_text(value))
    pmids = []
    for item in values:
        pmid = _to_text(item).strip()
        if pmid.isdigit() and pmid not in pmids:
            pmids.append(pmid)
    return sorted(pmids)

PUBMED_EVIDENCE_DETAIL_MARKERS = [
    "题名:",
    "作者:",
    "期刊:",
    "研究类型:",
    "治疗相关证据摘要:",
    "引用标注:",
    "DOI:",
    "链接:",
]

def _pubmed_evidence_detail_score(text: Any) -> int:
    raw = _to_text(text)
    if not raw.strip():
        return 0
    score = 0
    if "PMID:" in raw:
        score += 1
    if "【PubMed-" in raw or "PubMed-" in raw:
        score += 2
    score += sum(1 for marker in PUBMED_EVIDENCE_DETAIL_MARKERS if marker in raw)
    return score

def _has_detailed_pubmed_evidence(text: Any) -> bool:
    raw = _to_text(text)
    return bool(_extract_pmids(raw)) and _pubmed_evidence_detail_score(raw) >= 4

def _extract_cited_pmids(text: str) -> List[str]:
    text = _to_text(text)
    pmids = set(re.findall(r"\[PMID:\s*(\d+)\]", text))
    pmids.update(re.findall(r"(?<![A-Za-z])PMID\s*[:：]?\s*(\d+)", text))
    return sorted(pmids)

def _build_csco_rag_query(
    cancer_type: str,
    stage: str = "",
    t: str = "",
    n: str = "",
    m: str = "",
    risk_level: str = "",
) -> str:
    cancer_type = _to_text(cancer_type).strip().upper()
    base_query = CSCO_RAG_QUERY_BY_CANCER.get(
        cancer_type,
        f"{CANCER_NAME_CN.get(cancer_type, cancer_type)} CSCO 诊疗指南 治疗原则 推荐意见 治疗方案 适应证",
    )
    query_parts = [base_query, "CSCO 诊疗指南 治疗原则 推荐意见 治疗方案 适应证"]
    if stage:
        query_parts.append(f"临床分期:{stage}")
    if t or n or m:
        query_parts.append(f"TNM分期:{t}{n}{m}")
    if risk_level:
        query_parts.append(f"风险:{risk_level}")
    return " ".join(part for part in query_parts if _to_text(part).strip())

def _normalize_pubmed_citations(content: str, allowed_pmids: List[str]) -> str:
    allowed_set = set(allowed_pmids)

    def normalize_bracket(match):
        pmid = match.group(1)
        return f"[PMID: {pmid}]" if pmid in allowed_set else ""

    def normalize_plain(match):
        pmid = match.group(1)
        return f"[PMID: {pmid}]" if pmid in allowed_set else ""

    content = re.sub(r"\[PMID\s*[:：]\s*(\d+)\]", normalize_bracket, _to_text(content), flags=re.I)
    content = re.sub(r"(?<![\[\w])PMID\s*[:：]?\s*(\d+)", normalize_plain, content, flags=re.I)
    content = re.sub(r"\s+([；;，,。])", r"\1", content)
    return content

def _strip_markdown_fences(text: str) -> str:
    text = _remove_think_blocks(text)
    if text.startswith("```"):
        text = re.sub(r"^```(?:markdown|md)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()

def _normalize_csco_page(page: str) -> str:
    page = _to_text(page).strip()
    if "未知" in page:
        return "页码未知"
    match = re.search(r"(\d+)", page)
    return f"第{match.group(1)}页" if match else page

def _make_csco_citation(filename: str, page: str) -> str:
    filename = _to_text(filename).strip()
    page = _normalize_csco_page(page)
    return f"(CSCO指南，{filename}，{page})"

def _extract_allowed_csco_citations(guide_content: str) -> List[str]:
    guide_content = _to_text(guide_content)
    citations = []
    pattern = r"【CSCO-\d+\s*\|\s*([^|】]+?)\s*\|\s*([^】]+?)】"
    for filename, page in re.findall(pattern, guide_content):
        citation = _make_csco_citation(filename, page)
        if citation not in citations:
            citations.append(citation)
    return citations

def _normalize_csco_citation_text(text: str) -> str:
    text = _to_text(text)
    match = re.search(r"CSCO指南[，,]\s*([^，,()（）]+)[，,]\s*(第\s*\d+\s*页|页码未知)", text)
    if not match:
        return text.strip()
    return _make_csco_citation(match.group(1), match.group(2))

def _extract_cited_csco_citations(content: str) -> List[str]:
    content = _to_text(content)
    raw_citations = re.findall(r"[（(]\s*CSCO指南[，,][^）)]*[）)]", content)
    citations = []
    for raw in raw_citations:
        citation = _normalize_csco_citation_text(raw)
        if citation not in citations:
            citations.append(citation)
    return citations

def _normalize_plan_text(content: str, allowed_csco_citations: List[str]) -> str:
    content = _strip_markdown_fences(content)
    # 统一常见的半角括号、半角逗号和“第 15 页”这类写法。
    def repl(match):
        return _make_csco_citation(match.group(1), match.group(2))

    content = re.sub(
        r"[（(]\s*CSCO指南[，,]\s*([^，,()（）]+)[，,]\s*(第\s*\d+\s*页|页码未知)\s*[）)]",
        repl,
        content,
    )

    # 如果模型漏写文件名但页码能唯一匹配RAG清单，则自动补齐文件名。
    page_to_citation = {}
    for citation in allowed_csco_citations:
        page_match = re.search(r"[，,]\s*(第\d+页|页码未知)[）)]", citation)
        if page_match:
            page = page_match.group(1)
            page_to_citation.setdefault(page, set()).add(citation)
    unique_page_to_citation = {
        page: next(iter(citations))
        for page, citations in page_to_citation.items()
        if len(citations) == 1
    }

    def fill_missing_filename(match):
        page = _normalize_csco_page(match.group(1))
        return unique_page_to_citation.get(page, match.group(0))

    content = re.sub(
        r"[（(]\s*CSCO指南[，,]\s*(第\s*\d+\s*页|页码未知)\s*[）)]",
        fill_missing_filename,
        content,
    )
    return content.strip()

def _extract_markdown_section(text: str, header: str) -> str:
    match = re.search(rf"^\s*{re.escape(header)}\s*(.*?)(?=\n\s*## |\Z)", _to_text(text), flags=re.S | re.M)
    return match.group(1).strip() if match else ""

TREATMENT_TABLE_HEADERS = ["治疗阶段", "具体治疗方案", "剂量/给药/疗程", "适用依据", "证据标注"]
TREATMENT_TABLE_HEADER = "| " + " | ".join(TREATMENT_TABLE_HEADERS) + " |"

def _split_markdown_table_row(line: str) -> List[str]:
    text = _to_text(line).strip().replace("｜", "|")
    if not text.startswith("|"):
        return []
    return [cell.strip() for cell in text.strip("|").split("|")]

def _is_markdown_separator_row(cells: List[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)

def _compact_table_cells(cells: List[str]) -> List[str]:
    return [re.sub(r"\s+", "", cell) for cell in cells]

def _find_treatment_table_header_issues(section: str) -> List[str]:
    issues: List[str] = []
    expected_count = len(TREATMENT_TABLE_HEADERS)
    expected_compact = _compact_table_cells(TREATMENT_TABLE_HEADERS)
    candidates = []

    for line in _to_text(section).splitlines():
        cells = _split_markdown_table_row(line)
        if not cells or _is_markdown_separator_row(cells):
            continue
        compact_cells = _compact_table_cells(cells)
        if compact_cells[:expected_count] == expected_compact:
            return []
        score = sum(1 for header in expected_compact if header in compact_cells)
        if score > 0:
            candidates.append((score, line, cells, compact_cells))

    if not candidates:
        if "|" in _to_text(section) or "｜" in _to_text(section):
            return [
                "推荐治疗方案表格表头错误：未找到包含固定列名的Markdown表头；"
                f"必须逐字使用: {TREATMENT_TABLE_HEADER}"
            ]
        return [
            "推荐治疗方案表格格式错误：未找到Markdown表格表头；"
            f"必须逐字使用: {TREATMENT_TABLE_HEADER}"
        ]

    _, line, cells, compact_cells = max(candidates, key=lambda item: item[0])
    missing_headers = [
        header for header, compact_header in zip(TREATMENT_TABLE_HEADERS, expected_compact)
        if compact_header not in compact_cells
    ]
    extra_headers = [
        cell for cell, compact_cell in zip(cells, compact_cells)
        if compact_cell and compact_cell not in expected_compact
    ]

    if missing_headers:
        issues.append(
            "推荐治疗方案表格表头缺少固定列名: "
            + "、".join(missing_headers)
            + f"；必须逐字使用: {TREATMENT_TABLE_HEADER}"
        )
    if len(cells) != expected_count:
        issues.append(
            f"推荐治疗方案表格表头列数错误：固定表头为{expected_count}列，当前表头解析到{len(cells)}列；"
            f"当前表头: | {' | '.join(cells)} |"
        )
    if extra_headers:
        issues.append(
            "推荐治疗方案表格表头包含非固定列名或改写列名: "
            + "、".join(extra_headers)
        )
    if not missing_headers and compact_cells[:expected_count] != expected_compact:
        issues.append(
            "推荐治疗方案表格表头顺序错误：必须严格按顺序使用 "
            + " | ".join(TREATMENT_TABLE_HEADERS)
        )

    if not issues:
        issues.append(
            "推荐治疗方案表格表头格式错误："
            f"当前表头: | {' | '.join(cells)} |；必须逐字使用: {TREATMENT_TABLE_HEADER}"
        )
    issues.append(f"推荐治疗方案表格表头原始行: {line[:180]}")
    return issues

def _parse_treatment_table_rows(section: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    in_table = False
    for line in _to_text(section).splitlines():
        cells = _split_markdown_table_row(line)
        if not cells:
            if in_table:
                break
            continue
        compact_cells = _compact_table_cells(cells)
        if compact_cells[:len(TREATMENT_TABLE_HEADERS)] == TREATMENT_TABLE_HEADERS:
            in_table = True
            continue
        if not in_table or _is_markdown_separator_row(cells):
            continue
        if len(cells) >= len(TREATMENT_TABLE_HEADERS):
            rows.append(dict(zip(TREATMENT_TABLE_HEADERS, cells[:len(TREATMENT_TABLE_HEADERS)])))
    return rows

def _find_treatment_table_column_issues(section: str) -> List[str]:
    issues: List[str] = []
    in_table = False
    expected_count = len(TREATMENT_TABLE_HEADERS)
    data_row_index = 0
    for line in _to_text(section).splitlines():
        cells = _split_markdown_table_row(line)
        if not cells:
            if in_table:
                break
            continue
        compact_cells = _compact_table_cells(cells)
        if compact_cells[:expected_count] == TREATMENT_TABLE_HEADERS:
            in_table = True
            continue
        if not in_table or _is_markdown_separator_row(cells):
            continue
        data_row_index += 1
        if len(cells) != expected_count:
            issues.append(
                f"推荐治疗方案表格第{data_row_index}条治疗建议列数错误："
                f"固定表头为{expected_count}列，但该行解析到{len(cells)}列；"
                "必须单独填写“适用依据”列，且CSCO/PMID只能放在“证据标注”列。"
                f"原始行: {line[:180]}"
            )
    return issues

def _render_treatment_table(rows: List[Dict[str, str]]) -> str:
    lines = [TREATMENT_TABLE_HEADER, "|---|---|---|---|---|"]
    for row in rows:
        lines.append("| " + " | ".join(_to_text(row.get(header)).strip() for header in TREATMENT_TABLE_HEADERS) + " |")
    return "\n".join(lines)

def _debug_treatment_plan_output(label: str, content: str) -> None:
    """Print enough structure to diagnose why treatment-plan table parsing failed."""
    if not TREATMENT_PLAN_DEBUG:
        return

    content = _to_text(content)
    recommendation_section = _extract_markdown_section(content, "## 推荐治疗方案")
    treatment_rows = _parse_treatment_table_rows(recommendation_section)
    found_headers = re.findall(r"^\s*##\s+(.+?)\s*$", content, flags=re.M)
    compact_section = re.sub(r"\s+", "", recommendation_section)
    compact_header = re.sub(r"\s+", "", TREATMENT_TABLE_HEADER)
    pipe_lines = [
        line for line in recommendation_section.splitlines()
        if "|" in line or "｜" in line
    ][:8]
    preview = content[:TREATMENT_PLAN_DEBUG_CHARS]
    truncated = len(content) > TREATMENT_PLAN_DEBUG_CHARS

    print(f"\n[治疗方案LLM输出诊断] {label}", flush=True)
    print(f"- content_len: {len(content)}", flush=True)
    print(f"- has_title_##推荐治疗方案: {'## 推荐治疗方案' in content}", flush=True)
    print(f"- found_h2_headers: {json.dumps(found_headers, ensure_ascii=False)}", flush=True)
    print(f"- recommendation_section_len: {len(recommendation_section)}", flush=True)
    print(f"- has_required_table_header: {compact_header in compact_section}", flush=True)
    print(f"- parsed_treatment_rows: {len(treatment_rows)}", flush=True)
    if pipe_lines:
        print("- first_pipe_lines_in_recommendation_section:", flush=True)
        for line in pipe_lines:
            print(f"  {line}", flush=True)
    else:
        print("- first_pipe_lines_in_recommendation_section: <none>", flush=True)
    print("----- TREATMENT PLAN RAW OUTPUT START -----", flush=True)
    print(preview, flush=True)
    if truncated:
        print(
            f"...[已截断，仅显示前{TREATMENT_PLAN_DEBUG_CHARS}字符；"
            "可设置TREATMENT_PLAN_DEBUG_CHARS查看更多]",
            flush=True,
        )
    print("----- TREATMENT PLAN RAW OUTPUT END -----\n", flush=True)

def _normalize_treatment_phrase(text: str) -> str:
    raw = _to_text(text)
    raw = re.sub(r"[（(]\s*CSCO指南.*?[）)]", "", raw)
    raw = re.sub(r"[（(](.*?)[）)]", r"+\1", raw)
    raw = re.sub(r"\[[^\]]*\]", "", raw)
    raw = re.sub(r"剂量需按.*", "", raw)
    raw = re.sub(r"证据不足[，,、；;]?", "", raw)
    raw = re.sub(r"(具体药物|需|补充|关键检测|后再定|人工复核|建议|选择|治疗方案|治疗|方案|用药|剂量|给药|疗程)", "", raw)
    raw = raw.replace("（", "").replace("）", "").replace("(", "").replace(")", "")
    raw = re.sub(r"\s+", "", raw)
    return raw.strip("：:，,。；;、|")

TREATMENT_COMPONENT_STOPWORDS = {
    "治疗", "方案", "用药", "剂量", "给药", "疗程", "具体药物", "关键检测", "人工复核",
    "化疗", "免疫", "免疫治疗", "靶向", "靶向治疗", "内分泌", "内分泌治疗",
    "系统", "系统治疗", "辅助", "辅助治疗", "新辅助", "新辅助治疗",
    "维持", "维持治疗", "局部", "局部治疗", "综合", "综合策略", "综合治疗策略",
}
TREATMENT_CONNECTOR_PATTERN = r"[+＋、,，;；/]|联合|合并|序贯|同步|以及|和|及"

def _is_meaningful_treatment_component(phrase: str) -> bool:
    phrase = _to_text(phrase).strip()
    if len(phrase) < 2 or phrase in TREATMENT_COMPONENT_STOPWORDS:
        return False
    if len(phrase) <= 2 and phrase.endswith("疗"):
        return False
    return bool(re.search(r"[\u4e00-\u9fffA-Za-z0-9]", phrase))

def _extract_treatment_phrases(text: str) -> List[str]:
    raw = _to_text(text)
    phrases: List[str] = []
    normalized_raw = _normalize_treatment_phrase(raw)
    has_connector = bool(re.search(TREATMENT_CONNECTOR_PATTERN, normalized_raw))
    chunks = re.split(TREATMENT_CONNECTOR_PATTERN, normalized_raw)
    for chunk in chunks:
        phrase = _normalize_treatment_phrase(chunk)
        if (
            2 <= len(phrase) <= 40
            and _is_meaningful_treatment_component(phrase)
            and not any(phrase == existing for existing in phrases)
        ):
            phrases.append(phrase)
    if not phrases and not has_connector and _is_meaningful_treatment_component(normalized_raw):
        phrases.append(normalized_raw)
    if len(phrases) > 1:
        phrases = [
            phrase for phrase in phrases
            if not re.fullmatch(r"[A-Za-z0-9-]{1,5}", phrase)
        ] or phrases
    return phrases[:8]

def _pubmed_support_text_for_pmid(evidence_section: str, pmid: str) -> str:
    supports = []
    pmid_pattern = re.compile(rf"(?:\[PMID:\s*{re.escape(pmid)}\]|PMID\s*[:：]?\s*{re.escape(pmid)})", flags=re.I)
    for line in _to_text(evidence_section).splitlines():
        if pmid_pattern.search(line):
            match = re.search(r"支持(?:的)?建议\s*[:：]\s*(.+)$", line.strip())
            if match:
                supports.append(match.group(1).strip())
    return " ".join(supports).strip()

def _pubmed_support_matches_treatment(treatment: str, support_text: str) -> tuple[bool, str]:
    support_compact = _normalize_treatment_phrase(support_text)
    if not support_compact:
        return False, "证据来源说明中该PMID没有写明支持的建议"
    treatment_phrases = _extract_treatment_phrases(treatment)
    if treatment_phrases:
        missing_phrases = [phrase for phrase in treatment_phrases if phrase not in support_compact]
        if not missing_phrases:
            return True, ""
        return False, "证据来源说明未包含表格治疗方案组件: " + "、".join(missing_phrases[:5])
    treatment_compact = _normalize_treatment_phrase(treatment)
    if treatment_compact and treatment_compact in support_compact:
        return True, ""
    return False, "证据来源说明未包含表格中的具体治疗方案"

def _has_specific_treatment_name(text: str) -> bool:
    return bool(_extract_treatment_phrases(text))

def _valid_row_csco(row_csco: set[str], allowed_set: set[str]) -> set[str]:
    return row_csco & allowed_set if allowed_set else set(row_csco)

def _valid_row_pmids(row_pmids: set[str], available_pmids: set[str]) -> set[str]:
    return row_pmids & available_pmids

def _classify_evidence_row(row_csco: set[str], row_pmids: set[str], allowed_set: set[str], available_pmids: set[str]) -> str:
    has_csco = bool(_valid_row_csco(row_csco, allowed_set))
    has_pmid = bool(_valid_row_pmids(row_pmids, available_pmids))
    if has_csco and has_pmid:
        return "double"
    if has_csco:
        return "csco_only"
    if has_pmid:
        return "pubmed_only"
    return "invalid"

def _validate_evidence_mix(
    row_evidence_types: List[str],
    unique_csco: set[str],
    unique_pmids: set[str],
) -> List[str]:
    double_count = row_evidence_types.count("double")
    csco_only_count = row_evidence_types.count("csco_only")
    pubmed_only_count = row_evidence_types.count("pubmed_only")
    invalid_count = row_evidence_types.count("invalid")
    valid_row_count = double_count + csco_only_count + pubmed_only_count
    has_required_sources = len(unique_csco) >= 2 and len(unique_pmids) >= 2

    double_ok = double_count >= 2 and has_required_sources
    mixed_ok = (
        double_count >= 1
        and csco_only_count >= 1
        and pubmed_only_count >= 1
        and valid_row_count >= 3
        and has_required_sources
    )
    single_ok = csco_only_count >= 2 and pubmed_only_count >= 2 and has_required_sources
    if double_ok or mixed_ok or single_ok:
        return []

    return [
        "证据组合不合格："
        f"当前双证据{double_count}条、CSCO-only{csco_only_count}条、PubMed-only{pubmed_only_count}条、"
        f"无合法证据{invalid_count}条、合法治疗建议{valid_row_count}条、"
        f"不同CSCO证据来源{len(unique_csco)}个、不同PubMed PMID{len(unique_pmids)}个；"
        "需满足以下任一模式："
        "1) 全双证据：至少2条双证据治疗建议，且至少2个不同CSCO证据来源和2个不同PMID；"
        "2) 单双混合：至少1条双证据、1条CSCO-only、1条PubMed-only，总合法建议不少于3条，且至少2个不同CSCO证据来源和2个不同PMID；"
        "3) 全单证据：至少2条CSCO-only建议和2条PubMed-only建议，且至少2个不同CSCO证据来源和2个不同PMID。"
    ]

def _validate_treatment_plan_output(
    content: str,
    literature_results: str,
    allowed_csco_citations: Optional[List[str]] = None,
    available_pmids: Optional[List[str]] = None) -> List[str]:
    allowed_csco_citations = allowed_csco_citations or []
    available_pmids = set(_coerce_pmids(available_pmids) or _extract_pmids(literature_results))
    content = _normalize_plan_text(content, allowed_csco_citations)
    content = _normalize_pubmed_citations(content, sorted(available_pmids))
    literature_results = _to_text(literature_results)
    missing = []

    #检查治疗方案格式
    required_headers = ["## 推荐治疗方案", "## 治疗路线图", "## 注意事项", "## 证据来源说明"]
    absent_headers = [header for header in required_headers if header not in content]
    if absent_headers:
        missing.append(f"缺少方案内容: {', '.join(absent_headers)}")
    found_headers = re.findall(r"^\s*##\s+(.+?)\s*$", content, flags=re.M)
    expected_headers = [header.replace("## ", "") for header in required_headers]
    if found_headers and found_headers != expected_headers:
        missing.append(
            "二级标题必须严格且仅按顺序使用: " + " -> ".join(required_headers)
        )

    recommendation_section = _extract_markdown_section(content, "## 推荐治疗方案")
    evidence_section = _extract_markdown_section(content, "## 证据来源说明")
    required_table_header = TREATMENT_TABLE_HEADER
    compact_section = re.sub(r"\s+", "", recommendation_section).replace("｜", "|")
    compact_header = re.sub(r"\s+", "", required_table_header)
    header_issues: List[str] = []
    if compact_header not in compact_section:
        header_issues = _find_treatment_table_header_issues(recommendation_section)
        missing.extend(header_issues)
    treatment_rows = _parse_treatment_table_rows(recommendation_section)
    if len(treatment_rows) == 0:
        column_issues = _find_treatment_table_column_issues(recommendation_section)
        if column_issues:
            missing.extend(column_issues[:3])
        elif header_issues:
            pass
        else:
            missing.append("推荐治疗方案表格未解析到治疗建议行，请严格输出Markdown表格并至少生成符合证据组合规则的治疗建议")

    #检查PubMed引用情况
    if "PubMed" not in content and "[PMID:" not in content:
        missing.append("证据来源说明中缺少PubMed文献证据")

    if not available_pmids:
        missing.append("PubMed文献证据中未找到可引用PMID")
    cited_pmids = set(_extract_cited_pmids(content))
    if not cited_pmids:
        missing.append("推荐治疗方案中缺少[PMID: x]文献标注")
    invalid_pmids = cited_pmids - available_pmids
    if invalid_pmids:
        missing.append(f"引用了PubMed搜索结果中不存在的PMID: {', '.join(sorted(invalid_pmids))}")

    #检查CSCO指南引用情况：必须来自RAG返回的允许引用清单，不能编造页码或使用CSCO-n当页码。
    cited_csco = set(_extract_cited_csco_citations(content))
    if not cited_csco:
        missing.append("缺少规范CSCO引用，格式必须为(CSCO指南，文件名，第N页)或(CSCO指南，文件名，页码未知)")
    if allowed_csco_citations:
        allowed_set = set(allowed_csco_citations)
        invalid_csco = cited_csco - allowed_set
        if invalid_csco:
            missing.append(
                "CSCO引用不在RAG检索允许清单中，不能编造或改写: "
                + " | ".join(sorted(invalid_csco))
            )
    if re.search(r"CSCO-\d+", content):
        missing.append("CSCO引用中不能使用CSCO-n编号代替真实页码")

    allowed_set = set(allowed_csco_citations)
    row_evidence_types: List[str] = []
    unique_row_csco: set[str] = set()
    unique_row_pmids: set[str] = set()
    for idx, row in enumerate(treatment_rows, 1):
        stage_label = _to_text(row.get("治疗阶段")).strip()
        treatment = _to_text(row.get("具体治疗方案")).strip()
        applicability = _to_text(row.get("适用依据")).strip()
        evidence_tag = _to_text(row.get("证据标注")).strip()
        if not stage_label:
            missing.append(f"第{idx}条治疗建议缺少治疗阶段")
        elif re.search(r"双证据|CSCO-only|PubMed-only|证据优先|补充建议", stage_label, flags=re.I):
            missing.append(f"第{idx}条治疗建议的治疗阶段必须写临床阶段/治疗线别，不能写证据类型标签: {stage_label}")
        if not treatment:
            missing.append(f"第{idx}条治疗建议缺少具体治疗方案")
        elif "证据不足" not in treatment and not _has_specific_treatment_name(treatment):
            missing.append(f"第{idx}条治疗建议缺少具体药物/联合方案/局部治疗方式: {treatment[:80]}")
        if not applicability:
            missing.append(f"第{idx}条治疗建议缺少适用依据，必须说明本行治疗建议与患者Stage/TNM/风险等级或文献研究对象的匹配关系")
        elif not re.search(
            r"分期|TNM|风险|适应|适用|治疗线|患者|研究对象|术后|一线|二线|辅助|新辅助|复发|转移|不可切除|局部|手术|分子|检测|Stage|risk|line|patient|population|postoperative|adjuvant|neoadjuvant",
            applicability,
            flags=re.I,
        ):
            missing.append(f"第{idx}条治疗建议的适用依据过于笼统，需写明患者Stage/TNM/风险等级或PubMed文献研究对象/治疗场景: {applicability[:80]}")

        row_csco = set(_extract_cited_csco_citations(evidence_tag))
        row_pmids = set(_extract_cited_pmids(evidence_tag))
        valid_csco = _valid_row_csco(row_csco, allowed_set)
        valid_pmids = _valid_row_pmids(row_pmids, available_pmids)
        row_evidence_type = _classify_evidence_row(row_csco, row_pmids, allowed_set, available_pmids)
        row_evidence_types.append(row_evidence_type)
        unique_row_csco.update(valid_csco)
        unique_row_pmids.update(valid_pmids)
        if row_evidence_type == "invalid":
            missing.append(f"第{idx}条治疗建议的证据标注必须至少包含一个合法CSCO页码引用或一个合法PubMed PMID")
        invalid_row_csco = row_csco - allowed_set if allowed_set else set()
        if invalid_row_csco:
            missing.append(f"第{idx}条治疗建议引用了RAG允许清单外的CSCO证据: {' | '.join(sorted(invalid_row_csco))}")
        invalid_row_pmids = row_pmids - available_pmids
        if invalid_row_pmids:
            missing.append(f"第{idx}条治疗建议引用了PubMed搜索结果外的PMID: {', '.join(sorted(invalid_row_pmids))}")
        for pmid in sorted(valid_pmids):
            support_text = _pubmed_support_text_for_pmid(evidence_section, pmid)
            if not support_text:
                missing.append(f"证据来源说明缺少PMID {pmid}对应的PubMed支持建议")
                continue
            matches, reason = _pubmed_support_matches_treatment(treatment, support_text)
            if not matches:
                missing.append(
                    f"第{idx}条治疗建议与证据来源说明中PMID {pmid}支持的建议不一致: {reason}"
                )

    if treatment_rows:
        missing.extend(_validate_evidence_mix(row_evidence_types, unique_row_csco, unique_row_pmids))

    return missing

def _build_treatment_plan_template(
    allowed_csco_citations: List[str],
    available_pmids: List[str],
    stage: str = "",
    t: str = "",
    n: str = "",
    m: str = "",
    risk_level: str = "",
) -> str:
    csco_example = allowed_csco_citations[0] if allowed_csco_citations else "(CSCO指南，文件名，第N页)"
    csco_example_2 = allowed_csco_citations[1] if len(allowed_csco_citations) > 1 else csco_example
    pmid_example = f"[PMID: {available_pmids[0]}]" if available_pmids else "[PMID: XXXXXX]"
    pmid_example_2 = f"[PMID: {available_pmids[1]}]" if len(available_pmids) > 1 else pmid_example
    evidence_example = f"{csco_example}；{pmid_example}"
    tnm_text = "".join(part for part in [t, n, m] if _to_text(part).strip()) or "未提供"
    patient_basis_example = (
        f"临床分期:{stage or '未提供'}；TNM分期:{tnm_text}；风险等级:{risk_level or '未提供'}；"
        "说明本行治疗建议与CSCO检索片段的适应证/治疗线别或PubMed文献研究对象相符"
    )
    return f"""## 推荐治疗方案
    | 治疗阶段 | 具体治疗方案 | 剂量/给药/疗程 | 适用依据 | 证据标注 |
    |---|---|---|---|---|
    | 根据本行证据自行填写临床阶段/治疗线别 | 写出该证据支持的具体药物、联合方案或明确局部治疗方式；不能把非治疗证据写进推荐表 | 写出剂量/给药/疗程；未知则写“剂量需按指南/药品说明书和体表面积个体化” | {patient_basis_example} | {evidence_example} |

    ## 治疗路线图
    1. 初始评估：自己根据CSCO证据和文献证据生成完善的评估内容，例如病理、影像、TNM分期、体能状态和关键分子检测情况。
    2. 一线/新辅助/辅助治疗：自己根据CSCO证据和文献证据生成完善的治疗内容，例如治疗方案选择、药物选择、剂量/给药/疗程建议和适用依据。
    3. 疗效评估：自己根据CSCO证据和文献证据生成完善的评估内容，例如评估时间点、评估方法（影像学、实验室等）和评估内容（肿瘤大小变化、肿瘤标志物变化、症状改善等）。
    4. 后续治疗/维持/随访：自己根据CSCO证据和文献证据生成完善的预后计划，例如后续治疗方案选择、维持治疗建议和随访计划。

    ## 注意事项
    1. 用药前检查：自己根据CSCO证据和文献证据生成完善的检查内容，例如血常规、肝肾功能、心肺功能、感染筛查及必要分子检测等。
    2. 主要不良反应监测：自己根据CSCO证据和文献证据生成完善的监测内容，例如骨髓抑制、肝肾毒性、免疫相关不良反应和神经毒性等。
    3. 疗效评估时间点：自己根据CSCO证据和文献证据生成完善的评估时间点建议，例如通常每2-3个治疗周期结合影像和实验室结果评估一次。
    4. 缺失关键分子检测时的处理：自己根据CSCO证据和文献证据生成完善的处理建议，例如先补充分子检测或MDT讨论，避免无证据用药。

    ## 证据来源说明
    - CSCO证据: {csco_example}
    - CSCO证据: {csco_example_2}
    - PubMed证据: {pmid_example} 支持的建议：必须与推荐治疗方案表格中对应PMID所在行的具体治疗方案一致
    - PubMed证据: {pmid_example_2} 支持的建议：必须与推荐治疗方案表格中对应PMID所在行的具体治疗方案一致
    """

def _format_allowed_list(items: List[str]) -> str:
    if not items:
        return "- 无"
    return "\n".join(f"- {item}" for item in items)

def _build_treatment_retry_feedback(missing_items: List[str]) -> str:
    items = [_to_text(item).strip() for item in missing_items if _to_text(item).strip()]
    lines = [f"- {item}" for item in items]

    if any(("表格" in item or "未解析到治疗建议行" in item or "二级标题" in item) for item in items):
        lines.extend([
            "- 表格修复要求：只输出四个二级标题，不要输出前言、总结、代码块、HTML表格、截图说明、段落列表或额外标题。",
            f"- 表格修复要求：推荐治疗方案必须使用标准Markdown表格，表头必须逐字为：{TREATMENT_TABLE_HEADER}",
            "- 表格修复要求：表头下一行必须是分隔行“|---|---|---|---|---|”，其后每条治疗建议单独占一行。",
        ])

    if any(("证据组合不合格" in item or "CSCO-only" in item or "PubMed-only" in item or "不同CSCO证据来源" in item) for item in items):
        lines.extend([
            "- 证据组合修复要求：优先生成2条双证据治疗建议，每条同时包含一个允许的CSCO页码引用和一个允许的PubMed PMID。",
            "- 证据组合修复要求：如果双证据不足，生成1条双证据 + 1条CSCO-only + 1条PubMed-only。",
            "- 证据组合修复要求：如果仍不满足，生成2条CSCO-only + 2条PubMed-only；整体必须使用至少2个不同CSCO证据来源和2个不同PMID。",
        ])

    if any(("支持的建议" in item or "PMID" in item and "一致" in item) for item in items):
        lines.append("- PubMed修复要求：证据来源说明中每个使用过的PMID都必须写“支持的建议：...”，且核心治疗方案必须与表格中标注该PMID的具体治疗方案一致。")

    return "\n".join(lines)

def _repair_treatment_plan_output(
    content: str,
    literature_results: str,
    allowed_csco_citations: List[str],
    available_pmids: Optional[List[str]] = None,
) -> str:
    content = _normalize_plan_text(content, allowed_csco_citations)
    available_pmids = _coerce_pmids(available_pmids) or _extract_pmids(literature_results)
    default_csco = allowed_csco_citations[0] if allowed_csco_citations else ""
    content = _normalize_pubmed_citations(content, available_pmids)

    # 把模型写错的CSCO引用替换为RAG允许清单中的真实引用，避免因标点/文件名/页码格式漂移失败。
    def replace_csco(match):
        normalized = _normalize_csco_citation_text(match.group(0))
        return normalized if normalized in allowed_csco_citations else default_csco

    content = re.sub(r"[（(]\s*CSCO指南[，,][^）)]*[）)]", replace_csco, content)

    required_headers = ["## 推荐治疗方案", "## 治疗路线图", "## 注意事项", "## 证据来源说明"]
    sections = {}
    for header in required_headers:
        sections[header] = _extract_markdown_section(content, header) if header in content else ""

    if not sections["## 推荐治疗方案"]:
        sections["## 推荐治疗方案"] = content

    table_header = TREATMENT_TABLE_HEADER
    compact_rec = re.sub(r"\s+", "", sections["## 推荐治疗方案"])
    compact_header = re.sub(r"\s+", "", table_header)
    treatment_rows = _parse_treatment_table_rows(sections["## 推荐治疗方案"]) if compact_header in compact_rec else []
    if not treatment_rows:
        treatment_rows = []

    def build_repair_evidence_plan() -> List[tuple[str, str]]:
        csco_items = allowed_csco_citations[:2]
        pmid_items = available_pmids[:2]
        if len(csco_items) >= 2 and len(pmid_items) >= 2:
            return [(csco_items[0], pmid_items[0]), (csco_items[1], pmid_items[1])]
        if csco_items and pmid_items:
            plan = [(csco_items[0], pmid_items[0])]
            if len(csco_items) >= 2:
                plan.append((csco_items[1], ""))
            if len(pmid_items) >= 2:
                plan.append(("", pmid_items[1]))
            return plan
        if len(csco_items) >= 2:
            return [(csco_items[0], ""), (csco_items[1], "")]
        if len(pmid_items) >= 2:
            return [("", pmid_items[0]), ("", pmid_items[1])]
        if csco_items:
            return [(csco_items[0], "")]
        if pmid_items:
            return [("", pmid_items[0])]
        return []

    repair_evidence_plan = build_repair_evidence_plan()
    min_rows = len(repair_evidence_plan) or 1
    while len(treatment_rows) < min_rows:
        treatment_rows.append({
            "治疗阶段": "证据不足/MDT复核",
            "具体治疗方案": "证据不足，需基于CSCO指南、PubMed证据、TNM分期、风险分层和MDT讨论制定个体化方案；具体药物需补充关键检测后再定",
            "剂量/给药/疗程": "剂量需按指南/药品说明书和体表面积个体化",
            "适用依据": "患者分期、风险和可用证据",
            "证据标注": "",
        })
    treatment_rows = treatment_rows[:min_rows]

    for index, row in enumerate(treatment_rows):
        stage_label = _to_text(row.get("治疗阶段")).strip()
        if not stage_label or re.search(r"双证据|CSCO-only|PubMed-only|证据优先|补充建议", stage_label, flags=re.I):
            row["治疗阶段"] = "证据不足/MDT复核"
        if repair_evidence_plan:
            csco, pmid = repair_evidence_plan[index % len(repair_evidence_plan)]
            evidence_parts = []
            if csco:
                evidence_parts.append(csco)
            if pmid:
                evidence_parts.append(f"[PMID: {pmid}]")
            row["证据标注"] = "；".join(evidence_parts)
        for header in TREATMENT_TABLE_HEADERS:
            if not _to_text(row.get(header)).strip():
                row[header] = "证据不足，需人工复核"
    sections["## 推荐治疗方案"] = _render_treatment_table(treatment_rows)

    if not sections["## 治疗路线图"]:
        sections["## 治疗路线图"] = (
            "1. 初始评估：完善病理、影像、TNM分期、体能状态和关键分子检测。\n"
            "2. 一线/新辅助/辅助治疗：依据CSCO指南和PubMed证据进行MDT决策。\n"
            "3. 疗效评估：按治疗周期进行影像学、实验室和不良反应评估。\n"
            "4. 后续治疗/维持/随访：根据疗效、耐受性和复发风险调整方案。"
        )

    if not sections["## 注意事项"]:
        sections["## 注意事项"] = (
            "1. 用药前检查：血常规、肝肾功能、心肺功能、感染筛查及必要分子检测。\n"
            "2. 主要不良反应监测：骨髓抑制、肝肾毒性、免疫相关不良反应和神经毒性。\n"
            "3. 疗效评估时间点：通常每2-3个治疗周期结合影像和实验室结果评估。\n"
            "4. 缺失关键分子检测时的处理：先补充分子检测或MDT讨论，避免无证据用药。"
        )

    evidence_lines = []
    seen_csco = set()
    seen_pubmed = set()
    for row in treatment_rows:
        treatment = _to_text(row.get("具体治疗方案")).strip()
        evidence_cell = _to_text(row.get("证据标注"))
        for citation in _extract_cited_csco_citations(evidence_cell):
            if citation not in seen_csco:
                evidence_lines.append(f"- CSCO证据: {citation}")
                seen_csco.add(citation)
        for pmid in _extract_cited_pmids(evidence_cell):
            if pmid not in seen_pubmed:
                evidence_lines.append(f"- PubMed证据: [PMID: {pmid}] 支持的建议：{treatment}")
                seen_pubmed.add(pmid)
    sections["## 证据来源说明"] = "\n".join(evidence_lines).strip()

    return "\n\n".join(f"{header}\n{sections[header].strip()}" for header in required_headers).strip()

# ==================== 工具定义 ====================

@tool
def predict_tnm_stage(features_input: str, cancer_type: str = "BRCA") -> str:
    """使用TransMIL模型预测TNM分期。输入病理特征文件路径(.pt/.npy)"""
    try:
        features_path = Path(features_input)
        if not features_path.exists():
            return f"错误: 特征文件不存在: {features_input}"
        
        transmil = get_transmil_predictor(cancer_type)
        loaded_features = torch.load(features_path, map_location=transmil.device, weights_only=False)
        features = _unwrap_feature_tensor(loaded_features)
        print(f"[Pathomics特征] features shape: {features.shape if hasattr(features, 'shape') else type(features)}")
        coords, patch_size, coords_path = _load_clam_h5_coords(features_path)
        n_original_patches = None
        if hasattr(features, "shape") and len(features.shape) >= 2:
            n_original_patches = int(features.shape[1]) if len(features.shape) == 3 and int(features.shape[0]) == 1 else int(features.shape[0])
        if coords is not None and n_original_patches is not None and coords.shape[0] != n_original_patches:
            print(
                "  [TransMIL attention] coords count does not match feature count; "
                f"skip heatmaps: coords={coords.shape[0]}, features={n_original_patches}, h5={coords_path}"
            )
            coords = None

        attention_output_dir = features_path.parent.parent / "transmil_attention" / features_path.stem
        result = transmil.predict(
            features,
            return_attention=True,
            pathomics_coords=coords,
            pathomics_patch_size=patch_size,
            attention_output_dir=str(attention_output_dir),
            slide_id=features_path.stem,
        )
        
        t_stage = result.get('T', 'N/A')
        n_stage = result.get('N', 'N/A')
        m_stage = result.get('M', 'N/A')
        stage = result.get('Stage', 'N/A')
        print("[TransMIL预测结果] TNM、Stage分期结果见交互栏")
        output_lines = [
            f"T分期: {t_stage['label']}, 置信度: {t_stage['confidence']:.2f}",
            f"N分期: {n_stage['label']}, 置信度: {n_stage['confidence']:.2f}",
            f"M分期: {m_stage['label']}, 置信度: {m_stage['confidence']:.2f}",
            f"临床分期: {stage['label']}, 置信度: {stage['confidence']:.2f}",
        ]
        attention = result.get("attention") or {}
        heatmaps = attention.get("heatmaps") or {}
        if heatmaps:
            output_lines.append("\nTransMIL Gradient × Attention病理热力图:")
            for task_name in ("T", "N", "M", "Stage"):
                if heatmaps.get(task_name):
                    output_lines.append(f"  {task_name}: {heatmaps[task_name]}")
        score_modes = attention.get("score_modes") or {}
        if score_modes:
            output_lines.append(
                "\n1、Gradient × Attention计分模式: "
                + ", ".join(f"{task}={score_modes[task]}" for task in ("T", "N", "M", "Stage") if task in score_modes)
            )
            negative_tasks = [
                task for task in ("T", "N", "M", "Stage")
                if score_modes.get(task) == "negative_inhibition_raw"
            ]
            if negative_tasks:
                output_lines.append(
                    "(负向抑制热力图说明: "
                    + ", ".join(negative_tasks)
                    + " 颜色越深，抑制越强，表示颜色越深的区域越不需要关注。)"
                )
        csv_files = attention.get("csv_files") or {}
        if csv_files:
            output_lines.append("2、CSV权重分布文件:")
            if csv_files.get("cls_token_attention_csv"):
                output_lines.append(f"  CLS token attention map: {csv_files['cls_token_attention_csv']}")
            if csv_files.get("logits_gradients_csv"):
                output_lines.append(f"  各分类头logits梯度: {csv_files['logits_gradients_csv']}")
            if csv_files.get("attention_gradient_csv"):
                output_lines.append(f"  各分类头attention*gradient结果: {csv_files['attention_gradient_csv']}")
        if attention.get("attention_npz"):
            output_lines.append(f"3、Gradient × Attention原始数组: {attention['attention_npz']}")
        if attention.get("attention_summary_json"):
            output_lines.append(f"4、attention统计摘要: {attention['attention_summary_json']}")
        print("[Attention Map] TNM、Stage四分类任务的attention*gradient权重情况可见热力图\n" )
        return "\n".join(output_lines)
    except Exception as e:
        return f"TransMIL预测失败: {str(e)}"

def _unwrap_feature_tensor(loaded: Any) -> Any:
    if isinstance(loaded, dict):
        for key in ("features", "data"):
            if key in loaded:
                return loaded[key]
    return loaded

def _find_clam_h5_coords_path(features_path: Path) -> Path | None:
    stem = features_path.stem
    candidates = [
        features_path.parent.parent / "h5_files" / f"{stem}.h5",
        features_path.parent.parent.parent / "clam_output" / "patches" / f"{stem}.h5",
        features_path.with_suffix(".h5"),
    ]
    seen = set()
    for candidate in candidates:
        try:
            key = str(candidate.resolve())
        except Exception:
            key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None

def _load_clam_h5_coords(features_path: Path) -> tuple[np.ndarray | None, int | None, str | None]:
    h5_path = _find_clam_h5_coords_path(features_path)
    if h5_path is None:
        print(f"  [CMTA attention] CLAM h5 coords not found for {features_path}")
        return None, None, None

    try:
        import h5py

        with h5py.File(h5_path, "r") as handle:
            if "coords" not in handle:
                print(f"  [CMTA attention] coords dataset not found in {h5_path}")
                return None, None, str(h5_path)

            coords_dataset = handle["coords"]
            coords = np.asarray(coords_dataset)
            patch_size = coords_dataset.attrs.get("patch_size")
            if patch_size is None:
                patch_size = handle.attrs.get("patch_size")
            if patch_size is None:
                patch_size = handle.attrs.get("patch_size_level0")
            patch_size = int(np.asarray(patch_size).reshape(-1)[0]) if patch_size is not None else None
            return coords, patch_size, str(h5_path)
    except Exception as exc:
        print(f"  [CMTA attention] Failed to load CLAM h5 coords from {h5_path}: {exc}")
        return None, None, str(h5_path)


@tool
def predict_survival(features_input: str, gene_input: str, cancer_type: str = "BRCA") -> str:
    """使用CMTA模型预测生存时间，并输出跨模态attention统计与6组omic病理热力图。输入病理特征文件路径和基因组数据文件路径(.csv)"""
    try:
        missing_inputs = []
        if not features_input:
            missing_inputs.append("features_input(病理特征文件路径)")
        if not gene_input:
            missing_inputs.append("gene_input(基因组数据CSV路径)")
        if missing_inputs:
            message = f"[CMTA输入错误] 缺失必要输入: {', '.join(missing_inputs)}"
            print(message)
            raise ValueError(message)

        features_path = Path(features_input)
        gene_path = Path(gene_input)
        
        if not features_path.exists():
            message = f"[CMTA输入错误] 病理特征文件不存在: {features_input}"
            print(message)
            raise FileNotFoundError(message)
        if not gene_path.exists():
            message = f"[CMTA输入错误] 基因组数据文件不存在: {gene_input}"
            print(message)
            raise FileNotFoundError(message)
        
        cmta = get_predictor(cancer_type)

        features = _unwrap_feature_tensor(torch.load(features_path, map_location=cmta.device, weights_only=False))
        print(f"[Pathomics特征] features shape: {features.shape if hasattr(features, 'shape') else type(features)}")
        coords, patch_size, coords_path = _load_clam_h5_coords(features_path)
        n_original_patches = int(features.shape[0])
        if coords is not None and coords.shape[0] != n_original_patches:
            print(
                "  [CMTA attention] coords count does not match feature count; "
                f"skip heatmaps: coords={coords.shape[0]}, features={n_original_patches}, h5={coords_path}"
            )
            coords = None

        if n_original_patches > 4096:
            indices = np.random.choice(n_original_patches, 4096, replace=False)
            features = features[indices]
            if coords is not None:
                coords = coords[indices]
            print(f"[Pathomics特征OOM采样后] features shape: {features.shape}")

        attention_output_dir = features_path.parent.parent / "cmta_attention" / features_path.stem
        result = cmta.predict_survival(
            features,
            slide_id=features_path.stem,
            genomic_features=gene_input,
            return_attention=True,
            pathomics_coords=coords,
            pathomics_patch_size=patch_size,
            attention_output_dir=str(attention_output_dir),
        )
        
        survival_months = result.get('predicted_survival_months', 'N/A')
        risk_label = result.get('risk_level', 'N/A')
        risk_score = result.get('risk_score', 'N/A')
        print("[CMTA预测结果] 生存时间、风险等级和风险分数见交互栏")

        risk_score_text = f"{risk_score:.4f}" if isinstance(risk_score, (int, float)) else str(risk_score)
        output_lines = [
            f"预测生存时间: {survival_months:.2f}个月, 风险等级: {risk_label}, 风险分数: {risk_score_text}"
        ]
        attention = result.get("attention") or {}
        top_omic = (attention.get("p_in_g_summary") or {}).get("top_omic")
        if top_omic:
            fraction = float(top_omic.get("top_patch_fraction", 0.0)) * 100.0
            output_lines.append(
                "\n1、p_in_g_att最高频omic:\n "
                f"omic{top_omic.get('omic_index')} {top_omic.get('omic_name')} "
                f"({top_omic.get('top_patch_count')}/{(attention.get('p_in_g_summary') or {}).get('patch_count')} patches, "
                f"{fraction:.2f}%)"
            )
            top_omic_genes = attention.get("top_omic_genes") or {}
            genes = top_omic_genes.get("genes") or []
            if genes:
                output_lines.append(
                    "病理切片关注的基因组: "
                    f"omic{top_omic.get('omic_index')} {top_omic.get('omic_name')}; "
                    "基因: " + ", ".join(str(gene) for gene in genes)
                )
        heatmap_paths = attention.get("g_in_p_heatmaps") or []
        if attention.get("g_in_p_attention_csv"):
            output_lines.append(f"2、 g_in_p_att softmax attention CSV: {attention['g_in_p_attention_csv']}")
        if heatmap_paths:
            output_lines.append("3、g_in_p_att病理热力图: " + " ; ".join(heatmap_paths))
        if attention.get("attention_npz"):
            output_lines.append(f"4、cross-attention原始数组: {attention['attention_npz']}")
        if attention.get("attention_summary_json"):
            output_lines.append(f"5、attention统计摘要: {attention['attention_summary_json']}")

        print("[Attention Map] 各基因组的跨模态注意力权重情况可见病理热力图；对病理的跨模态注意力权重进行了统计，权重占比最高的基因组可见展示栏\n")
        return "\n".join(output_lines)
    except Exception as e:
        print(f"CMTA预测失败: {str(e)}")
        raise

@tool
def search_literature(
    cancer_type: str,
    stage: str = None,
    treatment: str = None,
    keywords: str = None,
    max_results: int = 8,
    recent_years: int = 5,
    min_impact_factor: float = 10.0) -> str:
    """搜索PubMed癌症治疗文献。默认返回近5年且期刊影响因子>=10的5-10篇证据。"""
    try:
        results = search_cancer_literature(
            cancer_type=cancer_type,
            stage=stage,
            treatment=treatment,
            keywords=keywords,
            max_results=max_results,
            recent_years=recent_years,
            min_impact_factor=min_impact_factor,
        )
        return format_pubmed_results(results)
    except Exception as e:
        return f"文献搜索失败: {str(e)}"

@tool
def generate_treatment_plan(cancer_type: str, t: str=None, n: str=None, m: str=None, stage: str = None,
                            risk_level: str = None, survival_months: str = None,
                            literature_results: str = None,
                            available_pmids: str = None) -> str:
    """生成治疗方案。输入癌症类型、临床分期、风险等级等信息"""
    try:
        global LAST_TREATMENT_CONTEXT
        cancer_type = _to_text(cancer_type).upper().strip()
        raw_required_inputs = {
            "T分期": t,
            "N分期": n,
            "M分期": m,
            "临床分期": stage,
            "预测生存时间": survival_months,
            "风险等级": risk_level,
        }
        missing_required_inputs = [
            label
            for label, value in raw_required_inputs.items()
            if not _to_text(value).strip()
            or _to_text(value).strip().lower() in {"n/a", "na", "none", "null"}
            or _to_text(value).strip() in {"无", "未提供", "未知"}
        ]
        if missing_required_inputs:
            raise ValueError(
                "生成治疗方案前缺少必要患者状态输入: "
                + "、".join(missing_required_inputs)
                + "。请先通过医学ReAct流程补充分期和生存风险信息。"
            )

        t = _normalize_tnm(t, "T")
        n = _normalize_tnm(n, "N")
        m = _normalize_tnm(m, "M")
        stage = _normalize_stage(stage)
        risk_level = _to_text(risk_level).strip()
        survival_months = _to_text(survival_months).strip()
        literature_results = _to_text(literature_results).strip()
        active_pubmed_evidence = get_active_pubmed_evidence()
        pubmed_evidence_content = active_pubmed_evidence
        if _pubmed_evidence_detail_score(literature_results) > _pubmed_evidence_detail_score(pubmed_evidence_content):
            pubmed_evidence_content = literature_results
        resolved_pmids = (
            _coerce_pmids(available_pmids)
            or _extract_pmids(literature_results)
            or _extract_pmids(pubmed_evidence_content)
        )

        if not literature_results or literature_results.strip() in {"无", "None", "未提供"}:
            raise ValueError("生成治疗方案前必须先调用search_literature，并传入PubMed文献证据。")
        if not resolved_pmids:
            raise ValueError("PubMed文献证据中未找到PMID，不能生成缺少文献标注的治疗方案。")
        if not _has_detailed_pubmed_evidence(pubmed_evidence_content):
            raise ValueError(
                "PubMed文献证据缺少题名、研究类型或治疗相关证据摘要，"
                "不能只基于PMID列表生成治疗方案。请先调用search_literature获取完整文献证据。"
            )

        rag = MedicalRAG(cancer_type)
        query = _build_csco_rag_query(
            cancer_type=cancer_type,
            stage=stage,
            t=t,
            n=n,
            m=m,
            risk_level=risk_level,
        )

        print(f"CSCO RAG检索查询: {query}")
        guide_content = rag.query(query, k=config.RAG_TOP_K)
        if not rag._has_valid_csco_rag(guide_content):
            raise ValueError(f"CSCO RAG检索失败，guide_content无效，不能生成只基于PubMed的治疗方案: {guide_content}")
        allowed_csco_citations = _extract_allowed_csco_citations(guide_content)
        if not allowed_csco_citations:
            raise ValueError(
                "CSCO RAG检索结果缺少可解析页码标记，不能生成可溯源治疗方案。"
                "请检查rag.py返回格式是否包含【CSCO-n | 文件名 | 第N页】。"
            )

        LAST_TREATMENT_CONTEXT = {
            "rag_query": query,
            "rag_retrieved": guide_content,
            "literature_results": pubmed_evidence_content,
            "literature_results_param": literature_results,
            "available_pmids": resolved_pmids,
            "allowed_csco_citations": allowed_csco_citations,
        }
        
        available_pmids_text = ", ".join(resolved_pmids) or "无可用PMID"
        allowed_csco_text = _format_allowed_list(allowed_csco_citations)
        output_template = _build_treatment_plan_template(
            allowed_csco_citations,
            resolved_pmids,
            stage=stage,
            t=t,
            n=n,
            m=m,
            risk_level=risk_level,
        )
        prompt = f"""你是一位资深肿瘤科医生，请根据以下信息生成个性化治疗方案：
        禁止凭空补充证据；禁止只写“化疗/内分泌治疗/免疫治疗/靶向治疗”等泛称。
        你必须严格复制下方模板结构，不能添加模板外标题、代码块、前言或总结。

        患者信息：
        - 癌症类型: {cancer_type}
        - T分期: {t or '未提供'}
        - N分期: {n or '未提供'}
        - M分期: {m or '未提供'}
        - 临床分期: {stage or '未提供'}
        - 风险等级: {risk_level or '未提供'}
        - 预测生存时间: {survival_months or '未提供'}

        CSCO诊疗指南相关内容（必须引用）：
        {guide_content}

        PubMed完整文献证据（必须结合，文献已按年份从近到远排列）：
        {pubmed_evidence_content}

        可引用的CSCO证据标注只能从以下清单中逐字复制，不能改写、不能自造、不能使用CSCO-n编号：
        {allowed_csco_text}

        可引用的PubMed PMID仅限这些: {available_pmids_text}

        输出必须严格只包含以下四个二级标题，标题名称和顺序不能改变：
        ## 推荐治疗方案
        ## 治疗路线图
        ## 注意事项
        ## 证据来源说明

        【推荐治疗方案】必须使用Markdown表格，字段固定为：
        | 治疗阶段 | 具体治疗方案 | 剂量/给药/疗程 | 适用依据 | 证据标注 |
        每一行都必须满足：
        - 每条治疗建议必须严格写满5列，顺序必须是“治疗阶段 | 具体治疗方案 | 剂量/给药/疗程 | 适用依据 | 证据标注”；禁止省略“适用依据”列，禁止把CSCO/PMID证据标注直接放进“适用依据”列。
        - “治疗阶段”必须由本行CSCO/PubMed证据内容自行决定，写证据真实支持的临床阶段或治疗线别，例如“一线治疗”“新辅助治疗”“辅助治疗”“术后辅助放疗”“维持治疗”“复发进展后二线治疗”等；不能固定照抄模板示例，不能写“双证据”“CSCO-only”“PubMed-only”等证据类型标签。
        - 不要把只支持检查、评估、MDT或风险分层的证据写进“推荐治疗方案”表；这些内容应写入“治疗路线图”或“注意事项”。推荐治疗方案表只写治疗、局部治疗、围手术期治疗、维持/后线治疗等证据支持的建议。
        - “适用依据”必须解释为什么本条治疗建议适用于当前患者：需要体现工具结果中的临床分期({stage or '未提供'})、TNM分期({(t or '') + (n or '') + (m or '') or '未提供'})、风险等级({risk_level or '未提供'})，并说明它与CSCO RAG检索片段中的适应证/治疗线别或PubMed文献中的研究对象、治疗场景相匹配。
        - 推荐治疗方案优先输出双证据建议；如果双证据不足，允许CSCO-only或PubMed-only补充建议。
        - 证据组合必须满足以下任一模式：
          1. 全双证据模式：至少2条治疗建议，每条同时含CSCO页码和PubMed PMID；
          2. 全单证据模式：至少2条CSCO-only建议和2条PubMed-only建议；
          3. 单双混合模式：至少1条双证据、1条CSCO-only、1条PubMed-only，且总建议不少于3条。
        - 整体至少使用2个不同CSCO证据来源和2个不同PubMed PMID；每行证据标注必须至少包含一个合法CSCO页码引用或一个合法PubMed PMID。
        - “具体治疗方案”必须写出具体药物、联合方案或明确的局部治疗方式；如果建议化疗/内分泌治疗/免疫治疗/靶向治疗，必须写出药物名或方案名，例如“吉西他滨+顺铂”“培美曲塞+卡铂”“奥希替尼”“他莫昔芬/芳香化酶抑制剂”等。
        - 如果CSCO或PubMed证据中没有足够信息支持具体药物，必须说明“证据不足，具体药物需补充关键检测后再定”，不能用笼统治疗替代具体方案。
        - “剂量/给药/疗程”尽量给出常用剂量、给药途径、周期或疗程；如果证据中没有剂量，写“剂量需按指南/药品说明书和体表面积个体化”，不能空缺。
        - “适用依据”必须是患者适配依据，不能只写证据编号；“证据标注”必须只写允许清单中的CSCO引用和/或真实PMID。
        - “证据标注”可以是“(CSCO指南，文件名，第N页)；[PMID: XXXXXX]”、单独CSCO引用或单独PubMed引用，但不能空缺。

        【治疗路线图】必须按时间顺序写：初始评估 -> 一线/新辅助/辅助治疗 -> 疗效评估 -> 后续治疗/维持/随访；必须结合预测生存时间用于预后判断、治疗强度、随访频率和MDT决策。

        【注意事项】必须包含：用药前检查、主要不良反应监测、疗效评估时间点、缺失关键分子检测时的处理；必须结合预测生存时间说明用药耐受性、手术适应症评估、术后并发症/不良反应监测和支持治疗。

        【证据来源说明】必须列出：
        - CSCO证据: 对应文件名和页码
        - PubMed证据: 使用过的PMID及其支持的建议；“支持的建议”必须与推荐治疗方案表格中标注同一PMID的“具体治疗方案”一致，必须写出相同的核心药物/联合方案/局部治疗方式

        【严格格式要求】每一条治疗建议必须标注来源：
        - 生成方案前必须完成CSCO RAG检索和PubMed文献检索；只能引用上方CSCO和PubMed检索结果中出现的内容
        - 引用CSCO指南时，必须从“可引用的CSCO证据标注清单”逐字复制完整括号内容，格式固定为"(CSCO指南，文件名，第N页)"或"(CSCO指南，文件名，页码未知)"
        - 禁止写"(CSCO指南，第N页)"，因为缺少文件名；禁止写"CSCO-1/CSCO-2"，因为那只是检索片段编号，不是页码
        - 引用PubMed文献内容时，必须使用对应文献的真实PMID标注"[PMID: XXXXXX]"，不得编造PMID
        - 每一条治疗建议必须至少有CSCO或PubMed之一支持；如果找不到同一治疗方案的双证据，可以使用单证据建议补充，但不要把支持其他治疗方案的PMID标到本行
        - 禁止在“推荐治疗方案”中写一种治疗方案，却在“证据来源说明”里把同一PMID解释为支持另一种治疗方案
        - 推荐治疗方案部分必须有Markdown表格分隔行"|---|---|---|---|---|"

        【必须复制并填充的输出模板】
        {output_template}"""
        
        llm = ChatZhipuAI(model=get_active_llm_model(), temperature=0.0)
        content = ""
        missing_items: List[str] = []
        quality_warnings: List[str] = []
        max_attempts = 3
        for attempt in range(max_attempts):
            if attempt == 0:
                current_prompt = prompt
            else:
                retry_feedback = _build_treatment_retry_feedback(missing_items)
                current_prompt = f"""{prompt}

                【上一版输出不合格，禁止沿用错误格式】
                {content}

                【上一版质检失败原因，必须逐条修复】
                {retry_feedback}

                请完全按“必须复制并填充的输出模板”重新生成。只输出四个二级标题及其内容；
                CSCO引用必须从允许清单逐字复制，PubMed引用必须从PMID清单中选择；
                每行治疗方案必须至少有CSCO或PubMed之一支持，优先双证据；证据来源说明中PMID支持的建议必须与表格具体治疗方案一致。"""

            response = _invoke_llm_with_retry(
                llm,
                current_prompt,
                call_name=f"generate_treatment_plan/attempt_{attempt + 1}",
            )
            content = response.content if hasattr(response, 'content') else str(response)
            content = _normalize_plan_text(content, allowed_csco_citations)
            content = _normalize_pubmed_citations(content, resolved_pmids)
            _debug_treatment_plan_output(f"attempt_{attempt + 1}", content)
            missing_items = _validate_treatment_plan_output(
                content,
                pubmed_evidence_content,
                allowed_csco_citations=allowed_csco_citations,
                available_pmids=resolved_pmids,
            )
            if not missing_items:
                break

            print(f"治疗方案第{attempt + 1}次生成不合格。缺失/问题:")
            for item in missing_items:
                print(f"  - {item}")

        if missing_items:
            print("治疗方案仍不合格，尝试自动修复格式和引用。缺失/问题:")
            for item in missing_items:
                print(f"  - {item}")
            repaired_content = _repair_treatment_plan_output(
                content,
                pubmed_evidence_content,
                allowed_csco_citations,
                available_pmids=resolved_pmids,
            )
            _debug_treatment_plan_output("auto_repair", repaired_content)
            repaired_missing = _validate_treatment_plan_output(
                repaired_content,
                pubmed_evidence_content,
                allowed_csco_citations=allowed_csco_citations,
                available_pmids=resolved_pmids,
            )
            content = repaired_content
            if repaired_missing:
                quality_warnings = repaired_missing
                print("治疗方案已自动修复格式和引用，但仍存在质检警告:")
                for item in quality_warnings:
                    print(f"  - {item}")
            else:
                missing_items = []

        return (
            "【CSCO RAG检索证据】\n"
            f"{guide_content}\n\n"
            "【PubMed文献证据】\n"
            f"{pubmed_evidence_content}\n\n"
            + (
                "【治疗方案质检警告】\n"
                + "\n".join(f"- {item}" for item in quality_warnings)
                + "\n\n"
                if quality_warnings else ""
            ) +
            "【最终治疗方案】\n"
            f"{content}"
        )
    except Exception as e:
        print(f"生成治疗方案失败: {str(e)}")
        raise


# ==================== 工具注册 ====================

TOOLS = [predict_tnm_stage, predict_survival, search_literature, generate_treatment_plan]

TOOL_MAP = {
    "predict_tnm_stage": predict_tnm_stage,
    "predict_survival": predict_survival,
    "search_literature": search_literature,
    "generate_treatment_plan": generate_treatment_plan,
}
