import {
  ArrowDown,
  BookOpenCheck,
  CheckCircle2,
  ChevronRight,
  ClipboardList,
  Clock3,
  Copy,
  Dna,
  FileArchive,
  FileText,
  Loader2,
  MessageSquarePlus,
  Microscope,
  MoreHorizontal,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRight,
  PanelRightClose,
  PanelRightOpen,
  Pencil,
  Plus,
  RefreshCw,
  Search,
  Send,
  Sparkles,
  Stethoscope,
  Trash2,
  X,
  ZoomIn,
  ZoomOut,
  XCircle
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { CSSProperties, PointerEvent as ReactPointerEvent, ReactNode, WheelEvent } from "react";
import { useEffect, useMemo, useRef, useState } from "react";

type UploadedFileMeta = {
  name: string;
  size: number;
  type: string;
};

type ChatMessage = {
  id: number;
  role: "user" | "assistant";
  content: string;
  analysisLog?: string;
};

type PatientSummary = Record<string, string>;

type WorkStage =
  | "idle"
  | "uploading-pathology"
  | "clam-running"
  | "pathology-ready"
  | "uploading-genome"
  | "genome-ready"
  | "agent-running"
  | "completed"
  | "failed";

type UploadPathologyResponse = {
  session_id: string;
  file_name: string;
};

type UploadGenomeResponse = {
  session_id: string;
  file_name: string;
  cancer_type: string;
  patient_summary: PatientSummary;
};

type ClamRunResponse = {
  session_id: string;
  pathology_jpg_url?: string;
  pathology_pt?: string;
};

type ChatResponse = {
  session_id: string;
  intent: string;
  cancer_type: string;
  llm_model?: string;
  result: string;
  literature?: string;
  analysis_log?: string;
};

type KnowledgeFile = {
  name: string;
  url: string;
  size?: number;
};

type KnowledgeFilesResponse = {
  files: KnowledgeFile[];
};

type KnowledgeUploadResponse = KnowledgeFilesResponse & {
  file: KnowledgeFile;
};

type LiteratureResponse = {
  session_id: string;
  has_literature: boolean;
  literature: string;
};

type LiteratureItem = {
  id: string;
  pmid: string;
  year: string;
  title: string;
  authors: string;
  journal: string;
  studyType: string;
  abstract: string;
  citation: string;
  doi: string;
  url: string;
  raw: string;
};

type CscoCitation = {
  key: string;
  id: string;
  filename: string;
  pageLabel: string;
  pageNumber: number | null;
  pdfUrl: string;
  imageUrl: string;
};

type CscoRagEvidenceItem = CscoCitation & {
  content: string;
};

type HeatmapItem = {
  label: string;
  sourcePath: string;
  url: string;
  mode?: string;
  description?: string;
};

type AnalysisVisualization = {
  id: string;
  type: "transmil" | "cmta";
  title: string;
  summary: string[];
  heatmaps: HeatmapItem[];
  genomicsFocus?: {
    omicLabel: string;
    genes: string[];
  };
};

type AnalysisFlowNodeKind = "thought" | "action" | "tool-flow" | "tool-result";
type AnalysisFlowNodeCta = "literature" | "csco-rag";
type AnalysisFlowNodeCtaPlacement = "title" | "after-csco-query";
type ExpandedSplitCard = "evidence" | "analysis" | null;

type AnalysisFlowNode = {
  id: string;
  kind: AnalysisFlowNodeKind;
  label: string;
  text: string;
  fullText: string;
  markdown?: boolean;
  cta?: AnalysisFlowNodeCta;
  ctaPlacement?: AnalysisFlowNodeCtaPlacement;
  ctaPayload?: string;
};

type AnalysisFlowRound = {
  id: string;
  roundNumber: number | null;
  nodes: AnalysisFlowNode[];
};

type ParsedAnalysisFlow = {
  taskText: string;
  rounds: AnalysisFlowRound[];
  finalText: string;
  finalMarkdown: boolean;
  fallbackText: string;
};

type CaseRecord = {
  id: string;
  title: string;
  customTitle?: string;
  sessionId: string | null;
  cancerType: string;
  llmModel: string;
  updatedAt: number;
  stage: WorkStage;
  messages: ChatMessage[];
  patientSummary: PatientSummary;
  wsiFile: UploadedFileMeta | null;
  genomeFile: UploadedFileMeta | null;
  genomeColumns?: string[];
  pathologyJpgUrl: string;
  pathologyPt: string;
  latestOutput: string;
  literatureText?: string;
};

const API_BASE_URL = (import.meta.env.VITE_MEDICAL_AGENT_API_BASE_URL || "").replace(/\/+$/, "");
const CASE_STORAGE_KEY = "medical-agent-case-records";
const DEFAULT_LLM_MODEL = "glm-4-flash";
const LEFT_SIDEBAR_WIDTH = 250;
const LEFT_SIDEBAR_COLLAPSED_WIDTH = 76;
const MAX_GENOME_COLUMN_PREVIEW = 18;
const GENOME_METADATA_COLUMNS = new Set(["case_id", "slide_id", "s_female", "is_female", "oncotree_code", "age"]);

function normalizeAnalysisText(text: string) {
  return (text || "").replace(/\r\n/g, "\n").replace(/\r/g, "\n");
}

function compactAnalysisText(text: string) {
  return normalizeAnalysisText(text)
    .replace(/={6,}/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function previewAnalysisText(text: string, _maxLength: number) {
  return displayAnalysisText(text);
}

function displayAnalysisText(text: string) {
  return normalizeAnalysisText(text)
    .replace(/^\s*={6,}\s*$/gm, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function getLastRegexGroup(text: string, pattern: RegExp) {
  const matches = Array.from(text.matchAll(pattern))
    .map((match) => (match[1] || "").trim())
    .filter(Boolean);
  return matches.length ? matches[matches.length - 1] : "";
}

function getFirstRegexGroup(text: string, pattern: RegExp) {
  return (text.match(pattern)?.[1] || "").trim();
}

function parseRoundNumber(value: string) {
  const trimmed = value.trim();
  if (/^\d+$/.test(trimmed)) return Number(trimmed);
  const digits: Record<string, number> = {
    "\u4e00": 1,
    "\u4e8c": 2,
    "\u4e09": 3,
    "\u56db": 4,
    "\u4e94": 5,
    "\u516d": 6,
    "\u4e03": 7,
    "\u516b": 8,
    "\u4e5d": 9,
    "\u5341": 10
  };
  if (trimmed === "\u5341") return 10;
  if (trimmed.includes("\u5341")) {
    const [tens, ones] = trimmed.split("\u5341");
    return (tens ? digits[tens] || 1 : 1) * 10 + (ones ? digits[ones] || 0 : 0);
  }
  return digits[trimmed] || 0;
}

function formatRoundLabel(roundNumber: number) {
  const digitLabels = ["", "\u4e00", "\u4e8c", "\u4e09", "\u56db", "\u4e94", "\u516d", "\u4e03", "\u516b", "\u4e5d"];
  const label =
    roundNumber <= 0
      ? String(roundNumber)
      : roundNumber < 10
        ? digitLabels[roundNumber]
        : roundNumber === 10
          ? "\u5341"
          : roundNumber < 20
            ? `\u5341${digitLabels[roundNumber % 10]}`
            : `${digitLabels[Math.floor(roundNumber / 10)]}\u5341${digitLabels[roundNumber % 10] || ""}`;
  return `\u7b2c${label}\u8f6e`;
}

function inferAnalysisTask(userTask: string, sourceText: string, finalAnswer: string) {
  const explicitTask = compactAnalysisText(userTask);
  if (explicitTask) return previewAnalysisText(explicitTask, 46);

  const combined = `${sourceText}\n${finalAnswer}`;
  if (/(generate_treatment_plan|medical_treatment_plan|治疗方案|个体化)/i.test(combined)) {
    return "生成个体化治疗方案";
  }
  if (/(predict_survival|生存|风险等级|风险预测)/i.test(combined)) {
    return "预测生存时间和风险等级";
  }
  if (/(predict_tnm_stage|TNM|临床分期|分期预测)/i.test(combined)) {
    return "预测TNM分期和临床分期";
  }
  return "医学任务分析";
}

function getAnalysisRoundChunks(text: string) {
  const roundPattern = /(?:^|\n)\s*(?:-{2,}\s*)?\u7b2c\s*([0-9\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+)\s*\u8f6e\s*(?:-{2,}|[:\uff1a])?/g;
  const matches = Array.from(text.matchAll(roundPattern));

  if (!matches.length) {
    const trimmed = text.trim();
    return trimmed ? [{ roundNumber: null, content: trimmed }] : [];
  }

  return matches.map((match, index) => {
    const start = match.index ?? 0;
    const nextStart = matches[index + 1]?.index ?? text.length;
    return {
      roundNumber: parseRoundNumber(match[1]),
      content: text.slice(start, nextStart).trim()
    };
  });
}

function stripEvidenceSections(text: string) {
  return normalizeAnalysisText(text)
    .replace(
      /\u3010CSCO\s*RAG\s*\u68c0\u7d22\u8bc1\u636e\u3011[\s\S]*?(?=\n\s*\u3010(?:PubMed\u6587\u732e\u8bc1\u636e|\u6cbb\u7597\u65b9\u6848\u8d28\u68c0\u8b66\u544a|\u6700\u7ec8\u6cbb\u7597\u65b9\u6848|\u6cbb\u7597\u65b9\u6848|\u5206\u6790\u6d41\u7a0b)[^\u3011]*\u3011|$)/g,
      ""
    )
    .replace(
      /\u3010PubMed\u6587\u732e\u8bc1\u636e\u3011[\s\S]*?(?=\n\s*\u3010(?:CSCO\s*RAG\s*\u68c0\u7d22\u8bc1\u636e|\u6cbb\u7597\u65b9\u6848\u8d28\u68c0\u8b66\u544a|\u6700\u7ec8\u6cbb\u7597\u65b9\u6848|\u6cbb\u7597\u65b9\u6848|\u5206\u6790\u6d41\u7a0b)[^\u3011]*\u3011|$)/g,
      ""
    )
    .trim();
}

function extractFinalTreatmentPlan(text: string) {
  const normalized = normalizeAnalysisText(text);
  const finalMatch = normalized.match(/\u3010\u6700\u7ec8\u6cbb\u7597\u65b9\u6848\u3011\s*([\s\S]*)/);
  if (finalMatch) {
    return stripEvidenceSections(`\u3010\u6700\u7ec8\u6cbb\u7597\u65b9\u6848\u3011\n${finalMatch[1]}`);
  }

  const recommendationMatch = normalized.match(/(^|\n)#{1,3}\s*\u63a8\u8350\u6cbb\u7597\u65b9\u6848[\s\S]*/);
  if (recommendationMatch) {
    return stripEvidenceSections(recommendationMatch[0]);
  }

  return "";
}

function isTreatmentAnalysis(text: string, finalAnswer: string) {
  return /(generate_treatment_plan|medical_treatment_plan|\u6700\u7ec8\u6cbb\u7597\u65b9\u6848|\u63a8\u8350\u6cbb\u7597\u65b9\u6848|\u6cbb\u7597\u65b9\u6848|\u3010CSCO\s*RAG\s*\u68c0\u7d22\u8bc1\u636e\u3011|\u3010PubMed\u6587\u732e\u8bc1\u636e\u3011)/i.test(
    `${text}\n${finalAnswer}`
  );
}

function isMarkdownLike(text: string) {
  return /(^|\n)\s*(#{1,6}\s+|\|.+\||[-*]\s+|\d+\.\s+)/.test(text);
}

function getConfidenceLabel(text: string) {
  return /\u7f6e\u4fe1\u533a\u95f4/.test(text) ? "\u7f6e\u4fe1\u533a\u95f4" : "\u7f6e\u4fe1\u5ea6";
}

function extractStagePredictionSummary(text: string) {
  const normalized = displayAnalysisText(text);
  const compact = normalized.replace(/\s+/g, " ");
  const confidenceLabel = getConfidenceLabel(normalized);
  const entries: Array<[string, RegExp]> = [
    ["T\u5206\u671f", /T\s*\u5206\u671f\s*[:\uff1a]\s*([^,\uff0c;\uff1b\n]+?)\s*[,，]\s*\u7f6e\u4fe1(?:\u5ea6|\u533a\u95f4)\s*[:\uff1a]\s*([0-9.%-]+)/i],
    ["N\u5206\u671f", /N\s*\u5206\u671f\s*[:\uff1a]\s*([^,\uff0c;\uff1b\n]+?)\s*[,，]\s*\u7f6e\u4fe1(?:\u5ea6|\u533a\u95f4)\s*[:\uff1a]\s*([0-9.%-]+)/i],
    ["M\u5206\u671f", /M\s*\u5206\u671f\s*[:\uff1a]\s*([^,\uff0c;\uff1b\n]+?)\s*[,，]\s*\u7f6e\u4fe1(?:\u5ea6|\u533a\u95f4)\s*[:\uff1a]\s*([0-9.%-]+)/i],
    [
      "\u4e34\u5e8a\u5206\u671f",
      /(?:\u4e34\u5e8a\u5206\u671f|Stage\s*\u5206\u671f)\s*[:\uff1a]\s*([^,\uff0c;\uff1b\n]+?)\s*[,，]\s*\u7f6e\u4fe1(?:\u5ea6|\u533a\u95f4)\s*[:\uff1a]\s*([0-9.%-]+)/i
    ]
  ];

  const lines = entries
    .map(([label, pattern]) => {
      const match = compact.match(pattern);
      return match ? `${label}: ${match[1].trim()}, ${confidenceLabel}: ${match[2].trim()}` : "";
    })
    .filter(Boolean);
  if (lines.length) return lines.join("\n");

  const fallbackLines = normalized
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .filter((line) => /(T|N|M)\s*\u5206\u671f|\u4e34\u5e8a\u5206\u671f|Stage\s*\u5206\u671f/i.test(line));
  return fallbackLines.join("\n");
}

function extractSurvivalPredictionSummary(text: string) {
  const normalized = displayAnalysisText(text);
  const compact = normalized.replace(/\s+/g, " ");
  const match = compact.match(
    /\u9884\u6d4b\u751f\u5b58\u65f6\u95f4\s*[:\uff1a]\s*([^,\uff0c;\uff1b\n]+?)\s*[,，]\s*\u98ce\u9669\u7b49\u7ea7\s*[:\uff1a]\s*([^,\uff0c;\uff1b\n]+?)\s*[,，]\s*\u98ce\u9669\u5206\u6570\s*[:\uff1a]\s*([-+0-9.]+)/i
  );
  if (match) {
    return `\u9884\u6d4b\u751f\u5b58\u65f6\u95f4: ${match[1].trim()}, \u98ce\u9669\u7b49\u7ea7: ${match[2].trim()}, \u98ce\u9669\u5206\u6570: ${match[3].trim()}`;
  }

  const fallbackLines = normalized
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .filter((line) => /\u9884\u6d4b\u751f\u5b58\u65f6\u95f4|\u98ce\u9669\u7b49\u7ea7|\u98ce\u9669\u5206\u6570/i.test(line));
  return fallbackLines.join("\n");
}

function summarizePredictionResult(text: string, action = "") {
  const combined = `${action}\n${text}`;
  if (/predict_tnm_stage/i.test(combined) || /(T|N|M)\s*\u5206\u671f\s*[:\uff1a]/i.test(combined)) {
    return extractStagePredictionSummary(text);
  }
  if (/predict_survival/i.test(combined) || /\u9884\u6d4b\u751f\u5b58\u65f6\u95f4\s*[:\uff1a]/.test(combined)) {
    return extractSurvivalPredictionSummary(text);
  }
  return "";
}

function extractPubmedReferenceLabels(text: string) {
  const refs = extractPubmedReferences(text);
  return refs.map((ref) => formatPubmedReferenceLabel(ref)).join("\n");
}

function extractPubmedReferences(text: string) {
  const normalized = normalizeAnalysisText(text);
  const seen = new Set<string>();
  const refs: Array<{ id: string; pmid: string; year: string }> = [];

  for (const match of normalized.matchAll(/【\s*PubMed-(\d+)\s*\|\s*PMID:\s*([^|】]+?)\s*\|\s*年份:\s*([^】]+?)\s*】/gi)) {
    const id = match[1].trim();
    const pmid = match[2].trim();
    const year = match[3].trim();
    const key = `${id}|${pmid}|${year}`;
    if (seen.has(key)) continue;
    seen.add(key);
    refs.push({ id, pmid, year });
  }

  return refs;
}

function formatPubmedReferenceLabel(ref: { id: string; pmid: string; year: string }) {
  return `【PubMed-${ref.id}|PMID:${ref.pmid}|年份:${ref.year}】`;
}

function extractPubmedEvidenceText(content: string) {
  const raw = normalizeAnalysisText(content);
  const match = raw.match(
    /【PubMed文献证据】\s*([\s\S]*?)(?=\n\s*【(?:治疗方案质检警告|最终治疗方案|治疗方案|分析流程|CSCO\s*RAG\s*检索证据)[^】]*】|$)/
  );
  return match?.[1]?.trim() || "";
}

function alignPubmedEvidenceToToolReferences(toolText: string, evidenceText: string) {
  const refs = extractPubmedReferences(toolText);
  if (!refs.length) return displayAnalysisText(evidenceText || toolText);

  const evidenceBlocks = normalizeAnalysisText(evidenceText)
    .split(/(?=【\s*PubMed-\d+\s*\|)/g)
    .map((block) => block.trim())
    .filter((block) => /^【\s*PubMed-\d+\s*\|/.test(block));
  const blockByPmid = new Map<string, string>();

  evidenceBlocks.forEach((block) => {
    const header = block.match(/^【\s*PubMed-\d+\s*\|\s*PMID:\s*([^|】]+?)(?:\s*\||\s*】)/i);
    const pmid = header?.[1]?.trim();
    if (pmid && !blockByPmid.has(pmid)) blockByPmid.set(pmid, block);
  });

  return refs
    .map((ref) => {
      const block = blockByPmid.get(ref.pmid);
      if (!block) return formatPubmedReferenceLabel(ref);
      return block.replace(/^【\s*PubMed-\d+\s*\|\s*PMID:\s*[^|】]+?\s*\|\s*年份:\s*[^】]+?\s*】/, formatPubmedReferenceLabel(ref));
    })
    .join("\n\n")
    .trim();
}

function extractLatestPubmedToolText(log: string) {
  const normalized = normalizeAnalysisText(log);
  const resultBlocks = Array.from(
    normalized.matchAll(
      /\u5de5\u5177\u7ed3\u679c\s*[:\uff1a]\s*([\s\S]*?)(?=\n\s*(?:={6,}|\u8bca\u65ad\u5b8c\u6210|---\s*\u7b2c\s*[0-9\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+\s*\u8f6e|\u7b2c\s*[0-9\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+\s*\u8f6e\s*[:\uff1a])|$)/g
    )
  )
    .map((match) => match[1] || "")
    .filter((block) => extractPubmedReferences(block).length);
  if (resultBlocks.length) return resultBlocks[resultBlocks.length - 1];

  const flowBlocks = Array.from(
    normalized.matchAll(/\u5de5\u5177\u5206\u6790\u6d41\u7a0b\s*[:\uff1a]\s*([\s\S]*?)(?=\n\s*\u5de5\u5177\u7ed3\u679c\s*[:\uff1a]|$)/g)
  )
    .map((match) => match[1] || "")
    .filter((block) => extractPubmedReferences(block).length);
  return flowBlocks.length ? flowBlocks[flowBlocks.length - 1] : "";
}

function isTreatmentToolText(action: string, text: string) {
  return /generate_treatment_plan|medical_treatment_plan/i.test(action) || /治疗方案LLM输出|【最终治疗方案】|##\s*推荐治疗方案/.test(text);
}

function isPubmedSearchToolText(action: string, text: string) {
  if (isTreatmentToolText(action, text)) return false;
  return /search_literature|pubmed(?:_search)?|literature/i.test(action) || /【\s*PubMed-\d+\s*\|/.test(text);
}

function formatPubmedToolTextForDisplay(text: string, action: string) {
  if (!isPubmedSearchToolText(action, text)) return "";
  return extractPubmedReferenceLabels(text);
}

function formatToolFlowForDisplay(toolFlow: string, action: string) {
  return formatPubmedToolTextForDisplay(toolFlow, action) || displayAnalysisText(toolFlow);
}

function formatToolResultForDisplay(toolResult: string, action: string, finalAnswer: string) {
  const predictionSummary = summarizePredictionResult(toolResult, action);
  if (predictionSummary) return predictionSummary;

  const treatmentTool = /generate_treatment_plan|medical_treatment_plan/i.test(action) || Boolean(extractFinalTreatmentPlan(toolResult));
  if (treatmentTool) {
    const finalPlan = extractFinalTreatmentPlan(toolResult) || extractFinalTreatmentPlan(finalAnswer);
    if (finalPlan) return finalPlan;
    return stripEvidenceSections(toolResult);
  }
  const pubmedSummary = formatPubmedToolTextForDisplay(toolResult, action);
  if (pubmedSummary) return pubmedSummary;
  return displayAnalysisText(toolResult);
}

function makeAnalysisFlowNode(
  kind: AnalysisFlowNodeKind,
  label: string,
  rawText: string,
  roundIndex: number,
  nodeIndex: number,
  markdown = false,
  cta?: AnalysisFlowNodeCta,
  ctaPlacement?: AnalysisFlowNodeCtaPlacement,
  ctaPayload?: string
): AnalysisFlowNode | null {
  const fullText = displayAnalysisText(rawText);
  if (!fullText) return null;

  return {
    id: `${roundIndex}-${nodeIndex}-${kind}`,
    kind,
    label,
    text: fullText,
    fullText,
    markdown,
    cta,
    ctaPlacement,
    ctaPayload
  };
}

function parseAnalysisFlow(log: string, userTask: string, finalAnswer: string): ParsedAnalysisFlow {
  const sourceText = normalizeAnalysisText(log);
  const finalFromLog = getLastRegexGroup(
    sourceText,
    /Final Answer\s*[:\uff1a]\s*([\s\S]*?)(?=\n\s*(?:={6,}|\u8bca\u65ad\u5b8c\u6210|---\s*\u7b2c\s*[0-9\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+\s*\u8f6e|\u7b2c\s*[0-9\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+\s*\u8f6e\s*[:\uff1a])|$)/gi
  );
  const taskText = inferAnalysisTask(userTask, sourceText, finalAnswer);
  const chunks = getAnalysisRoundChunks(sourceText);
  const rounds: AnalysisFlowRound[] = [];
  const treatmentFlow = isTreatmentAnalysis(sourceText, finalAnswer);

  chunks.forEach((chunk, roundIndex) => {
    let nodeIndex = 0;
    const nodes: AnalysisFlowNode[] = [];
    const thought = getFirstRegexGroup(
      chunk.content,
      /(?:^|\n)\s*Thought\s*[:\uff1a]\s*([\s\S]*?)(?=\n\s*(?:Action|Action Input|Observation|Final Answer)\s*[:\uff1a]|\n\s*\u8c03\u7528\u5de5\u5177\s*[:\uff1a]|\n\s*\u5de5\u5177\u5206\u6790\u6d41\u7a0b\s*[:\uff1a]|\n\s*\u5de5\u5177\u7ed3\u679c\s*[:\uff1a]|$)/i
    );
    const action =
      getFirstRegexGroup(chunk.content, /(?:^|\n)\s*Action\s*[:\uff1a]\s*([^\n]+)/i) ||
      getFirstRegexGroup(chunk.content, /(?:^|\n)\s*\u8c03\u7528\u5de5\u5177\s*[:\uff1a]\s*([^\n]+)/);
    const toolFlow = getFirstRegexGroup(
      chunk.content,
      /\u5de5\u5177\u5206\u6790\u6d41\u7a0b\s*[:\uff1a]\s*([\s\S]*?)(?=\n\s*\u5de5\u5177\u7ed3\u679c\s*[:\uff1a]|$)/
    );
    const rawToolResult = getFirstRegexGroup(
      chunk.content,
      /\u5de5\u5177\u7ed3\u679c\s*[:\uff1a]\s*([\s\S]*?)(?=\n\s*(?:={6,}|\u8bca\u65ad\u5b8c\u6210|---\s*\u7b2c\s*[0-9\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+\s*\u8f6e|\u7b2c\s*[0-9\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+\s*\u8f6e\s*[:\uff1a])|$)/
    );
    const formattedToolFlow = formatToolFlowForDisplay(toolFlow, action);
    const toolResult = formatToolResultForDisplay(rawToolResult, action, finalAnswer);
    const renderToolFlowMarkdown = treatmentFlow && isTreatmentToolText(action, formattedToolFlow) && isMarkdownLike(formattedToolFlow);
    const renderToolResultMarkdown = treatmentFlow && isMarkdownLike(toolResult);
    const toolFlowCta = /CSCO\s*RAG检索查询\s*[:：]/.test(formattedToolFlow) ? "csco-rag" : undefined;
    const toolResultCta = formatPubmedToolTextForDisplay(rawToolResult, action) ? "literature" : undefined;

    const addNode = (
      kind: AnalysisFlowNodeKind,
      label: string,
      rawText: string,
      markdown = false,
      cta?: AnalysisFlowNodeCta,
      ctaPlacement?: AnalysisFlowNodeCtaPlacement,
      ctaPayload?: string
    ) => {
      const node = makeAnalysisFlowNode(kind, label, rawText, roundIndex, nodeIndex, markdown, cta, ctaPlacement, ctaPayload);
      nodeIndex += 1;
      if (node) nodes.push(node);
    };

    addNode("thought", "Thought:", thought);
    addNode("action", "Action:", action.replace(/\s*\.\.\.\s*$/, ""));
    addNode("tool-flow", "工具分析流程:", formattedToolFlow, renderToolFlowMarkdown, toolFlowCta, "after-csco-query");
    addNode("tool-result", "工具结果:", toolResult, renderToolResultMarkdown, toolResultCta, "title", rawToolResult);

    if (nodes.length) {
      rounds.push({
        id: `round-${chunk.roundNumber ?? roundIndex + 1}`,
        roundNumber: chunk.roundNumber,
        nodes
      });
    }
  });

  const rawFinalAnswer = finalFromLog || finalAnswer;
  const finalPredictionSummary = treatmentFlow ? "" : summarizePredictionResult(rawFinalAnswer, sourceText);
  const finalCandidate = treatmentFlow
    ? extractFinalTreatmentPlan(finalFromLog) || extractFinalTreatmentPlan(finalAnswer) || stripEvidenceSections(finalFromLog || finalAnswer)
    : finalPredictionSummary || displayAnalysisText(rawFinalAnswer);

  return {
    taskText,
    rounds,
    finalText: finalCandidate,
    finalMarkdown: treatmentFlow && isMarkdownLike(finalCandidate),
    fallbackText: displayAnalysisText(sourceText)
  };
}

const llmModelOptions = [
  { value: "glm-4-flash", label: "GLM-4-flash" },
  { value: "glm-4.7-flash", label: "GLM-4.7-flash" },
  { value: "glm-z1-flash", label: "GLM-Z1-flash" },
  { value: "glm-4-plus", label: "GLM-4-plus" },
  { value: "glm-4-air", label: "GLM-4-air" }
];

const stageLabels: Record<WorkStage, string> = {
  idle: "等待输入",
  "uploading-pathology": "病理上传中",
  "clam-running": "CLAM 处理中",
  "pathology-ready": "病理特征已生成",
  "uploading-genome": "基因组上传中",
  "genome-ready": "病例数据已就绪",
  "agent-running": "智能体分析中",
  completed: "分析完成",
  failed: "处理失败"
};

const quickTasks: Array<{ title: string; desc: string; prompt: string; icon: LucideIcon }> = [
  {
    title: "分期预测",
    desc: "调用 TransMIL 预测 TNM 与临床分期",
    prompt: "请根据已上传的病理特征预测 TNM 分期和临床分期。",
    icon: Microscope
  },
  {
    title: "生存时间预测",
    desc: "融合病理特征与基因组数据估计生存风险",
    prompt: "请结合病理特征和基因组数据预测生存时间和风险等级。",
    icon: Dna
  },
  {
    title: "治疗方案预测",
    desc: "结合分期、生存风险、CSCO 指南和 PubMed 证据",
    prompt: "请结合分期、生存风险、CSCO 指南和 PubMed 文献生成可解释的治疗方案。",
    icon: BookOpenCheck
  }
];

let messageSeed = 0;

function nextMessageId() {
  messageSeed += 1;
  return Date.now() + messageSeed;
}

function createLocalCaseId() {
  return `case-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function apiUrl(path: string) {
  return `${API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
}

function absoluteApiUrl(path?: string) {
  if (!path) return "";
  if (/^https?:\/\//i.test(path)) return path;
  return apiUrl(path);
}

function fileToMeta(file: File): UploadedFileMeta {
  return {
    name: file.name,
    size: file.size,
    type: file.type || file.name.split(".").pop()?.toUpperCase() || "file"
  };
}

function formatTime(timestamp: number) {
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  }).format(timestamp);
}

function getCaseTitle(patientSummary: PatientSummary, wsiFile: UploadedFileMeta | null, genomeFile: UploadedFileMeta | null) {
  const caseId = patientSummary["Case ID"] || patientSummary.case_id || patientSummary["病例 ID"];
  if (caseId) return String(caseId);
  if (genomeFile) return genomeFile.name.replace(/\.[^.]+$/, "");
  if (wsiFile) return wsiFile.name.replace(/\.[^.]+$/, "");
  return "未命名病例";
}

function formatFileSize(size?: number) {
  if (!size || size < 0) return "PDF";
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / 1024 / 1024).toFixed(1)} MB`;
}

function extractLiteratureField(raw: string, label: string) {
  const fieldNames = "题名|作者|期刊|研究类型|治疗相关证据摘要|引用标注|DOI|链接";
  const match = raw.match(new RegExp(`${label}:\\s*([\\s\\S]*?)(?=\\n(?:${fieldNames}):|$)`));
  return match?.[1]?.trim() || "";
}

function parseLiteratureItems(text: string): LiteratureItem[] {
  const normalized = text.replace(/\r\n/g, "\n").replace(/\r/g, "\n").trim();
  if (!normalized || normalized.startsWith("未找到") || normalized.startsWith("搜索出错")) return [];

  const blocks = normalized
    .split(/(?=【PubMed-\d+\s*\|)/g)
    .map((block) => block.trim())
    .filter((block) => block.startsWith("【PubMed-"));
  if (!blocks.length) return [];

  return blocks.map((raw, index) => {
    const header = raw.match(/【PubMed-(\d+)\s*\|\s*PMID:\s*([^|]+)\|\s*年份:\s*([^】]+)】/);
    const pmid = header?.[2]?.trim() || "";
    const rawUrl = extractLiteratureField(raw, "链接");
    const url = /^https?:\/\//i.test(rawUrl) ? rawUrl : "";
    return {
      id: header?.[1]?.trim() || String(index + 1),
      pmid,
      year: header?.[3]?.trim() || "",
      title: extractLiteratureField(raw, "题名") || "未命名文献",
      authors: extractLiteratureField(raw, "作者"),
      journal: extractLiteratureField(raw, "期刊"),
      studyType: extractLiteratureField(raw, "研究类型"),
      abstract: extractLiteratureField(raw, "治疗相关证据摘要"),
      citation: extractLiteratureField(raw, "引用标注") || (pmid ? `[PMID: ${pmid}]` : ""),
      doi: extractLiteratureField(raw, "DOI"),
      url,
      raw
    };
  });
}

function removePubmedEvidenceSections(content: string) {
  return content.replace(
    /【PubMed文献证据】[\s\S]*?(?=\n\s*【(?:治疗方案质检警告|最终治疗方案|治疗方案|分析流程|CSCO\s*RAG\s*检索证据)[^】]*】|$)/g,
    ""
  );
}

function removeCscoRagEvidenceSections(content: string) {
  return content.replace(
    /【CSCO\s*RAG\s*检索证据】[\s\S]*?(?=\n\s*【(?:PubMed文献证据|治疗方案质检警告|最终治疗方案|治疗方案|分析流程)[^】]*】|$)/g,
    ""
  );
}

function getTreatmentCitationScope(content: string) {
  const cleaned = removeCscoRagEvidenceSections(removePubmedEvidenceSections(content));
  const finalPlanMatch = cleaned.match(/【最终治疗方案】\s*([\s\S]*)$/);
  return finalPlanMatch?.[1]?.trim() || cleaned;
}

function getReferencedLiteratureMarkers(content: string) {
  const citationScope = getTreatmentCitationScope(content);
  const pubmedIds = new Set<string>();
  const pmids = new Set<string>();

  for (const match of citationScope.matchAll(/(?:【|\[|\(|\b)PubMed-(\d+)/gi)) {
    pubmedIds.add(match[1]);
  }
  for (const match of citationScope.matchAll(/PMID[:：]?\s*(\d{5,})/gi)) {
    pmids.add(match[1]);
  }

  return { pubmedIds, pmids };
}

function filterPubmedEvidenceSections(content: string) {
  const { pubmedIds, pmids } = getReferencedLiteratureMarkers(content);
  const hasReferences = pubmedIds.size > 0 || pmids.size > 0;

  return content
    .replace(
      /(【PubMed文献证据】\s*)([\s\S]*?)(?=\n\s*【(?:治疗方案质检警告|最终治疗方案|治疗方案|分析流程|CSCO\s*RAG\s*检索证据)[^】]*】|$)/g,
      (_section, title: string, body: string) => {
        if (!hasReferences) return "";
        const blocks = body
          .split(/(?=【PubMed-\d+\s*\|)/g)
          .map((block) => block.trim())
          .filter((block) => block.startsWith("【PubMed-"));
        const referencedBlocks = blocks.filter((block) => {
          const header = block.match(/【PubMed-(\d+)\s*\|\s*PMID:\s*([^|】]+)(?:\||】)/);
          const id = header?.[1]?.trim() || "";
          const pmid = header?.[2]?.trim() || "";
          return pubmedIds.has(id) || (pmid && pmids.has(pmid));
        });

        return referencedBlocks.length ? `${title}${referencedBlocks.join("\n\n")}\n` : "";
      }
    )
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function cscoPdfUrl(filename: string, pageNumber: number | null) {
  const base = absoluteApiUrl(`/api/knowledge/csco/${encodeURIComponent(filename)}`);
  return pageNumber ? `${base}#page=${pageNumber}&toolbar=0&navpanes=0&view=FitH` : base;
}

function cscoPageImageUrl(filename: string, pageNumber: number | null) {
  if (!pageNumber) return "";
  return absoluteApiUrl(`/api/knowledge/csco/${encodeURIComponent(filename)}/pages/${pageNumber}`);
}

function parseCscoCitations(text: string): CscoCitation[] {
  const raw = text.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  const citations: CscoCitation[] = [];
  const seen = new Set<string>();

  const addCitation = (id: string, filename: string, pageText: string) => {
    const cleanFilename = filename.trim();
    const pageMatch = pageText.match(/(\d+)/);
    const pageNumber = pageMatch ? Number(pageMatch[1]) : null;
    const pageLabel = pageNumber ? `第${pageNumber}页` : pageText.trim() || "页码未知";
    const key = `${cleanFilename}::${pageLabel}`;
    if (!cleanFilename || seen.has(key)) return;
    seen.add(key);
    citations.push({
      key,
      id: id || String(citations.length + 1),
      filename: cleanFilename,
      pageLabel,
      pageNumber,
      pdfUrl: cscoPdfUrl(cleanFilename, pageNumber),
      imageUrl: cscoPageImageUrl(cleanFilename, pageNumber)
    });
  };

  for (const match of raw.matchAll(/【CSCO-(\d+)\s*\|\s*([^|】]+?\.pdf)\s*\|\s*([^】]+?)】/g)) {
    addCitation(match[1], match[2], match[3]);
  }

  for (const match of raw.matchAll(/[（(]CSCO指南[，,]\s*([^，,）)]+?\.pdf)[，,]\s*(第\d+页|页码未知)[）)]/g)) {
    addCitation(String(citations.length + 1), match[1], match[2]);
  }

  return citations;
}

function parseCscoRagEvidenceItems(text: string): CscoRagEvidenceItem[] {
  const raw = normalizeAnalysisText(text).trim();
  if (!raw) return [];

  const items: CscoRagEvidenceItem[] = [];
  const seen = new Set<string>();
  const blocks = raw
    .split(/(?=【CSCO-\d+\s*\|)/g)
    .map((block) => block.trim())
    .filter((block) => block.startsWith("【CSCO-"));

  blocks.forEach((block, index) => {
    const match = block.match(/^【CSCO-(\d+)\s*\|\s*([^|】]+?)\s*\|\s*([^】]+?)】\s*([\s\S]*)$/);
    if (!match) return;

    const id = match[1].trim() || String(index + 1);
    const filename = match[2].trim();
    const pageLabel = match[3].trim() || "页码未知";
    const content = displayAnalysisText(match[4]);
    const pageMatch = pageLabel.match(/(\d+)/);
    const pageNumber = pageMatch ? Number(pageMatch[1]) : null;
    const key = `${filename}::${pageLabel}::${id}`;
    if (!filename || seen.has(key)) return;
    seen.add(key);
    items.push({
      key,
      id,
      filename,
      pageLabel,
      pageNumber,
      pdfUrl: cscoPdfUrl(filename, pageNumber),
      imageUrl: cscoPageImageUrl(filename, pageNumber),
      content
    });
  });

  return items;
}

function extractCscoRagEvidenceText(content: string) {
  const raw = content.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  const match = raw.match(
    /【CSCO\s*RAG\s*检索证据】\s*([\s\S]*?)(?=\n\s*【(?:PubMed文献证据|治疗方案质检警告|最终治疗方案|治疗方案|分析流程)[^】]*】|$)/
  );
  return match?.[1]?.trim() || "";
}

function parseCscoRagCitations(content: string) {
  return parseCscoCitations(extractCscoRagEvidenceText(content));
}

function getReferencedCscoCitations(content: string) {
  return parseCscoCitations(getTreatmentCitationScope(content));
}

function formatCscoCitationLine(citation: CscoCitation) {
  return `【CSCO-${citation.id} | ${citation.filename} | ${citation.pageLabel}】`;
}

function filterReferencedCscoCitations(content: string) {
  const referenced = getReferencedCscoCitations(content);
  if (!referenced.length) return [];

  const referencedKeys = new Set(referenced.map((citation) => citation.key));
  const ragCitations = parseCscoRagCitations(content);
  if (!ragCitations.length) return referenced;

  const filtered = ragCitations.filter((citation) => referencedKeys.has(citation.key));
  return filtered.length ? filtered : referenced;
}

function filterCscoRagEvidenceSections(content: string) {
  const referencedKeys = new Set(getReferencedCscoCitations(content).map((citation) => citation.key));

  return content
    .replace(
      /(【CSCO\s*RAG\s*检索证据】\s*)([\s\S]*?)(?=\n\s*【(?:PubMed文献证据|治疗方案质检警告|最终治疗方案|治疗方案|分析流程)[^】]*】|$)/g,
      (_section, title: string, body: string) => {
        if (!referencedKeys.size) return "";
        const referencedRagCitations = parseCscoCitations(body).filter((citation) => referencedKeys.has(citation.key));
        return referencedRagCitations.length
          ? `${title}${referencedRagCitations.map(formatCscoCitationLine).join("\n")}\n`
          : "";
      }
    )
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function filterReferencedLiteratureItems(items: LiteratureItem[], content: string) {
  if (!items.length) return [];
  const { pubmedIds, pmids } = getReferencedLiteratureMarkers(content);

  if (!pubmedIds.size && !pmids.size) {
    return [];
  }

  const referenced = items.filter((item) => pubmedIds.has(item.id) || (item.pmid && pmids.has(item.pmid)));
  return referenced;
}

function encodePathSegments(path: string) {
  return path
    .replace(/\\/g, "/")
    .split("/")
    .filter(Boolean)
    .map((part) => encodeURIComponent(part))
    .join("/");
}

function filePathToSessionUrl(path: string, activeSessionId: string | null) {
  if (!path) return "";
  if (/^https?:\/\//i.test(path)) return path;
  if (!activeSessionId) return "";

  const normalized = path.trim().replace(/^["'`]+|["'`)]+$/g, "");
  const lowerPath = normalized.toLowerCase();
  const lowerSession = activeSessionId.toLowerCase();
  const sessionIndex = lowerPath.lastIndexOf(lowerSession);
  let relativePath = "";

  if (sessionIndex >= 0) {
    relativePath = normalized.slice(sessionIndex + activeSessionId.length).replace(/^[\\/]+/, "");
  } else if (!/^[A-Za-z]:[\\/]/.test(normalized) && !normalized.startsWith("/") && !normalized.startsWith("\\")) {
    relativePath = normalized;
  }

  if (!relativePath) return "";
  return absoluteApiUrl(`/api/files/${activeSessionId}/${encodePathSegments(relativePath)}`);
}

function getFileNameFromPath(path: string) {
  const parts = path.replace(/\\/g, "/").split("/");
  return parts[parts.length - 1] || path;
}

type HeatmapViewState = {
  scale: number;
  offsetX: number;
  offsetY: number;
};

type ImageDisplaySize = {
  width: number;
  height: number;
};

function clampOffset(value: number, max: number) {
  return Math.min(max, Math.max(-max, value));
}

function getContainedImageSize(frame: HTMLElement, image: HTMLImageElement): ImageDisplaySize {
  const frameRect = frame.getBoundingClientRect();
  const naturalWidth = image.naturalWidth || frameRect.width;
  const naturalHeight = image.naturalHeight || frameRect.height;
  const scale = Math.min(frameRect.width / naturalWidth, frameRect.height / naturalHeight);

  return {
    width: Math.max(1, Math.floor(naturalWidth * scale)),
    height: Math.max(1, Math.floor(naturalHeight * scale))
  };
}

function constrainHeatmapView(
  view: HeatmapViewState,
  frame: HTMLElement | null,
  image: HTMLImageElement | null
): HeatmapViewState {
  if (!frame || !image || view.scale <= 1) {
    return { scale: 1, offsetX: 0, offsetY: 0 };
  }

  const frameRect = frame.getBoundingClientRect();
  const baseWidth = image.offsetWidth || frameRect.width;
  const baseHeight = image.offsetHeight || frameRect.height;
  const maxOffsetX = Math.max(0, (baseWidth * view.scale - frameRect.width) / 2);
  const maxOffsetY = Math.max(0, (baseHeight * view.scale - frameRect.height) / 2);

  return {
    scale: view.scale,
    offsetX: clampOffset(view.offsetX, maxOffsetX),
    offsetY: clampOffset(view.offsetY, maxOffsetY)
  };
}

function HeatmapPreview({ heatmap, onOpen }: { heatmap: HeatmapItem; onOpen: (heatmap: HeatmapItem) => void }) {
  if (!heatmap.url) {
    return <div className="heatmap-missing">无法预览</div>;
  }

  return (
    <button
      type="button"
      className="heatmap-preview-trigger"
      aria-label={`打开 ${heatmap.label} 热力图`}
      title="点击查看大图"
      onClick={() => onOpen(heatmap)}
    >
      <img src={heatmap.url} alt={`${heatmap.label} heatmap`} draggable={false} />
      <span className="heatmap-zoom-icon" aria-hidden="true">
        <ZoomIn size={14} />
      </span>
    </button>
  );
}

function ZoomableImageLightbox({
  title,
  subtitle,
  imageUrl,
  alt,
  onClose,
  dialogLabel
}: {
  title: string;
  subtitle?: string;
  imageUrl: string;
  alt: string;
  onClose: () => void;
  dialogLabel: string;
}) {
  const frameRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const dragRef = useRef<{ pointerId: number; startX: number; startY: number; offsetX: number; offsetY: number } | null>(null);
  const [dragging, setDragging] = useState(false);
  const [view, setView] = useState<HeatmapViewState>({ scale: 1, offsetX: 0, offsetY: 0 });
  const [displaySize, setDisplaySize] = useState<ImageDisplaySize>({ width: 0, height: 0 });

  useEffect(() => {
    dragRef.current = null;
    setDragging(false);
    setView({ scale: 1, offsetX: 0, offsetY: 0 });
    setDisplaySize({ width: 0, height: 0 });
  }, [imageUrl]);

  const updateDisplaySize = () => {
    const frame = frameRef.current;
    const image = imageRef.current;
    if (!frame || !image || !image.naturalWidth || !image.naturalHeight) return;

    const nextSize = getContainedImageSize(frame, image);
    setDisplaySize((current) =>
      current.width === nextSize.width && current.height === nextSize.height ? current : nextSize
    );
    setView((current) => constrainHeatmapView(current, frame, image));
  };

  useEffect(() => {
    const frame = frameRef.current;
    if (!frame) return;

    const observer = new ResizeObserver(updateDisplaySize);
    observer.observe(frame);
    window.requestAnimationFrame(updateDisplaySize);
    return () => observer.disconnect();
  }, [imageUrl]);

  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [onClose]);

  const handleImageWheel = (event: WheelEvent<HTMLDivElement>) => {
    event.preventDefault();
    const frame = event.currentTarget;
    const frameRect = frame.getBoundingClientRect();
    const cursorX = event.clientX - frameRect.left - frameRect.width / 2;
    const cursorY = event.clientY - frameRect.top - frameRect.height / 2;
    const scaleFactor = event.deltaY < 0 ? 1.16 : 1 / 1.16;

    setView((current) => {
      const nextScale = Math.min(6, Math.max(1, current.scale * scaleFactor));
      if (nextScale === 1) return { scale: 1, offsetX: 0, offsetY: 0 };

      const scaleRatio = nextScale / current.scale;
      return constrainHeatmapView(
        {
          scale: nextScale,
          offsetX: current.offsetX + (cursorX - current.offsetX) * (1 - scaleRatio),
          offsetY: current.offsetY + (cursorY - current.offsetY) * (1 - scaleRatio)
        },
        frame,
        imageRef.current
      );
    });
  };

  const handlePointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    event.preventDefault();
    dragRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      offsetX: view.offsetX,
      offsetY: view.offsetY
    };
    setDragging(true);
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const handlePointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag) return;
    event.preventDefault();
    const frame = event.currentTarget;
    setView((current) =>
      constrainHeatmapView(
        {
          ...current,
          offsetX: drag.offsetX + event.clientX - drag.startX,
          offsetY: drag.offsetY + event.clientY - drag.startY
        },
        frame,
        imageRef.current
      )
    );
  };

  const handlePointerEnd = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (dragRef.current?.pointerId === event.pointerId) {
      dragRef.current = null;
      setDragging(false);
    }
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  const resetZoom = () => {
    setView({ scale: 1, offsetX: 0, offsetY: 0 });
  };

  return (
    <div className="heatmap-lightbox" role="dialog" aria-modal="true" aria-label={dialogLabel}>
      <div className="heatmap-lightbox-panel">
        <div className="heatmap-lightbox-header">
          <div>
            <strong>{title}</strong>
            {subtitle ? <small title={subtitle}>{subtitle}</small> : null}
          </div>
          <button type="button" className="heatmap-lightbox-close" aria-label="关闭图片" title="关闭图片" onClick={onClose}>
            <X size={18} />
          </button>
        </div>
        <div
          ref={frameRef}
          role="button"
          tabIndex={0}
          className={`heatmap-lightbox-image ${view.scale > 1 ? "is-zoomed" : ""} ${dragging ? "is-dragging" : ""}`}
          aria-label={dialogLabel}
          title="按住拖动图片，滚轮缩放"
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerEnd}
          onPointerCancel={handlePointerEnd}
          onDoubleClick={resetZoom}
          onWheel={handleImageWheel}
        >
          <img
            ref={imageRef}
            src={imageUrl}
            alt={alt}
            draggable={false}
            onLoad={updateDisplaySize}
            style={{
              width: displaySize.width ? `${displaySize.width}px` : undefined,
              height: displaySize.height ? `${displaySize.height}px` : undefined,
              transform: `translate(${view.offsetX}px, ${view.offsetY}px) scale(${view.scale})`,
              transformOrigin: "center center"
            }}
          />
          <span className="heatmap-zoom-icon heatmap-lightbox-zoom-icon" aria-hidden="true">
            {view.scale > 1 ? <ZoomOut size={16} /> : <ZoomIn size={16} />}
          </span>
        </div>
      </div>
    </div>
  );
}

function HeatmapLightbox({ heatmap, onClose }: { heatmap: HeatmapItem; onClose: () => void }) {
  return (
    <ZoomableImageLightbox
      title={heatmap.label}
      subtitle={getFileNameFromPath(heatmap.sourcePath)}
      imageUrl={heatmap.url}
      alt={`${heatmap.label} heatmap`}
      dialogLabel={`${heatmap.label} 热力图预览`}
      onClose={onClose}
    />
  );
}

function PathologyMaskPreview({ imageUrl, onOpen }: { imageUrl: string; onOpen: () => void }) {
  return (
    <button
      type="button"
      className="pathology-preview-trigger"
      aria-label="打开 CLAM mask 图像预览"
      title="点击查看大图"
      onClick={onOpen}
    >
      <img src={imageUrl} alt="CLAM tissue mask" draggable={false} />
      <span className="heatmap-zoom-icon pathology-zoom-icon" aria-hidden="true">
        <ZoomIn size={14} />
      </span>
    </button>
  );
}

function normalizeExtractedPath(path: string) {
  return path.trim().replace(/^["'`(]+/, "").replace(/[)"'`.,;，；。]+$/g, "");
}

function extractPngPaths(line: string) {
  const matches = line.match(/[A-Za-z]:[\\/].+?\.png|[\\/].+?\.png|[^\s;；,，]+?\.png/gi) || [];
  return [...new Set(matches.map(normalizeExtractedPath).filter(Boolean))];
}

function inferTransmilHeatmapLabel(line: string, path: string, index: number) {
  const taskMatch = line.match(/^\s*(T|N|M|Stage)\s*[:：]/i);
  if (taskMatch) {
    return taskMatch[1].toLowerCase() === "stage" ? "Stage" : taskMatch[1].toUpperCase();
  }

  const fileName = getFileNameFromPath(path).toLowerCase();
  if (/_t_.*heatmap|_t_grad_attention/.test(fileName)) return "T";
  if (/_n_.*heatmap|_n_grad_attention/.test(fileName)) return "N";
  if (/_m_.*heatmap|_m_grad_attention/.test(fileName)) return "M";
  if (/stage.*heatmap|stage_grad_attention/.test(fileName)) return "Stage";
  return ["T", "N", "M", "Stage"][index] || `Heatmap ${index + 1}`;
}

function inferCmtaHeatmapLabel(path: string, index: number) {
  const fileName = getFileNameFromPath(path);
  const fullMatch = fileName.match(/omic(\d+)_([^\\/]+?)_g_in_p_heatmap/i);
  if (fullMatch) return `omic${fullMatch[1]} ${fullMatch[2].replace(/_/g, " ")}`;

  const omicMatch = fileName.match(/omic(\d+)/i);
  if (omicMatch) return `omic${omicMatch[1]}`;
  return `omic${index + 1}`;
}

function parseTransmilScoreModes(line: string) {
  const modes: Record<string, string> = {};
  const matches = line.matchAll(/\b(T|N|M|Stage)\s*=\s*([a-z_]+)/gi);
  for (const match of matches) {
    const task = match[1].toLowerCase() === "stage" ? "Stage" : match[1].toUpperCase();
    modes[task] = match[2];
  }
  return modes;
}

function describeTransmilScoreMode(mode?: string) {
  if (!mode) return undefined;
  if (/negative|inhibition/i.test(mode)) return "负向模式：分数越低，颜色越深，表示越需要关注";
  if (/positive/i.test(mode)) return "正向模式：分数越高，颜色越深，表示越需要关注";
  if (/neutral|zero/i.test(mode)) return "中性权重图";
  return undefined;
}

function formatOmicSummary(line: string) {
  const trimmed = line.trim();
  const omicMatch = trimmed.match(/omic\s*([1-6])\s+(.+?)(?:\s*\(|$)/i);
  if (omicMatch) return `病理切片相关 omic: omic${omicMatch[1]} ${omicMatch[2].trim()}`;
  return trimmed.replace(/^p_in_g_att\s*/i, "病理切片相关 omic ");
}

function parseGeneList(text: string) {
  return text
    .split(/[,，、;；\s]+/)
    .map((item) => item.trim())
    .filter(Boolean)
    .filter((item, index, array) => array.findIndex((value) => value.toLowerCase() === item.toLowerCase()) === index)
    .slice(0, 18);
}

function splitCsvLine(line: string) {
  const cells: string[] = [];
  let current = "";
  let inQuotes = false;

  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    const next = line[index + 1];

    if (char === '"' && inQuotes && next === '"') {
      current += '"';
      index += 1;
    } else if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === "," && !inQuotes) {
      cells.push(current.trim());
      current = "";
    } else {
      current += char;
    }
  }

  cells.push(current.trim());
  return cells.map((cell) => cell.replace(/^"|"$/g, "").trim()).filter(Boolean);
}

function parseCsvHeaderColumns(text: string) {
  const firstLine = text
    .replace(/^\uFEFF/, "")
    .split(/\r?\n/)
    .find((line) => line.trim());
  if (!firstLine) return [];
  return splitCsvLine(firstLine);
}

async function readGenomeColumnPreview(file: File) {
  try {
    const headerText = await file.slice(0, 512 * 1024).text();
    const columns = parseCsvHeaderColumns(headerText);
    const featureColumns = columns.filter((column) => !GENOME_METADATA_COLUMNS.has(column.trim().toLowerCase()));
    return (featureColumns.length ? featureColumns : columns).slice(0, MAX_GENOME_COLUMN_PREVIEW);
  } catch {
    return [];
  }
}

function parseCmtaGenomicsFocus(line: string) {
  const trimmed = line.trim();
  if (!/(病理切片关注的基因组|p_in_g_att|最高频omic|相关.*omic)/i.test(trimmed) || !/omic/i.test(trimmed)) {
    return null;
  }

  const omicMatch = trimmed.match(/omic\s*([1-6])\s+(.+?)(?:\s*[;(（]|$)/i);
  const fallbackMatch = trimmed.match(/omic\s*([1-6])/i);
  const omicLabel = omicMatch
    ? `omic${omicMatch[1]} ${omicMatch[2].trim()}`
    : fallbackMatch
      ? `omic${fallbackMatch[1]}`
      : trimmed;

  const genesMatch = trimmed.match(/(?:基因(?:名)?|genes?)\s*[:：]\s*(.+)$/i);
  return {
    omicLabel,
    genes: genesMatch ? parseGeneList(genesMatch[1]) : []
  };
}

function createHeatmapItem(
  label: string,
  path: string,
  activeSessionId: string | null,
  mode?: string
): HeatmapItem {
  const sourcePath = normalizeExtractedPath(path);
  return {
    label,
    sourcePath,
    url: filePathToSessionUrl(sourcePath, activeSessionId),
    mode,
    description: describeTransmilScoreMode(mode)
  };
}

function appendHeatmapItem(items: HeatmapItem[], item: HeatmapItem) {
  if (!items.some((existing) => existing.sourcePath.toLowerCase() === item.sourcePath.toLowerCase())) {
    items.push(item);
  }
}

function parseAnalysisVisualizations(content: string, activeSessionId: string | null): AnalysisVisualization[] {
  const lines = content.replace(/\r\n/g, "\n").replace(/\r/g, "\n").split("\n");
  const transmilHeatmaps: HeatmapItem[] = [];
  const cmtaHeatmaps: HeatmapItem[] = [];
  const cmtaSummary: string[] = [];
  const transmilScoreModes: Record<string, string> = {};
  let cmtaGenomicsFocus: AnalysisVisualization["genomicsFocus"];

  lines.forEach((line) => {
    const paths = extractPngPaths(line);
    const lowerLine = line.toLowerCase();

    if (!paths.length) {
      Object.assign(transmilScoreModes, parseTransmilScoreModes(line));
      const genomicsFocus = parseCmtaGenomicsFocus(line);
      if (genomicsFocus) {
        cmtaGenomicsFocus = {
          omicLabel: genomicsFocus.omicLabel,
          genes: genomicsFocus.genes.length ? genomicsFocus.genes : cmtaGenomicsFocus?.genes || []
        };
      }
      if (/(p_in_g_att|top_omic|highest.*omic|最高频omic|相关.*omic)/i.test(line) && /omic/i.test(line)) {
        const summary = formatOmicSummary(line);
        if (!cmtaSummary.includes(summary)) cmtaSummary.push(summary);
      }
      return;
    }

    paths.forEach((path) => {
      const lowerPath = path.toLowerCase();
      const combined = `${lowerLine} ${lowerPath}`;
      if (combined.includes("transmil") || combined.includes("grad_attention") || /^\s*(T|N|M|Stage)\s*[:：]/i.test(line)) {
        const label = inferTransmilHeatmapLabel(line, path, transmilHeatmaps.length);
        appendHeatmapItem(
          transmilHeatmaps,
          createHeatmapItem(label, path, activeSessionId, transmilScoreModes[label])
        );
      } else if (combined.includes("cmta") || combined.includes("g_in_p") || combined.includes("omic")) {
        appendHeatmapItem(
          cmtaHeatmaps,
          createHeatmapItem(inferCmtaHeatmapLabel(path, cmtaHeatmaps.length), path, activeSessionId)
        );
      }
    });
  });

  const annotatedTransmilHeatmaps = transmilHeatmaps.map((heatmap) => {
    const mode = transmilScoreModes[heatmap.label] || heatmap.mode;
    return {
      ...heatmap,
      mode,
      description: describeTransmilScoreMode(mode)
    };
  });

  const visualizations: AnalysisVisualization[] = [];
  if (annotatedTransmilHeatmaps.length) {
    visualizations.push({
      id: "transmil",
      type: "transmil",
      title: "TransMIL 分类关注区域",
      summary: [],
      heatmaps: annotatedTransmilHeatmaps
    });
  }
  if (cmtaHeatmaps.length) {
    visualizations.push({
      id: "cmta",
      type: "cmta",
      title: "CMTA omic 关注区域",
      summary: cmtaSummary,
      heatmaps: cmtaHeatmaps,
      genomicsFocus: cmtaGenomicsFocus
    });
  }
  return visualizations;
}

function collapseCscoEvidenceSnippets(content: string) {
  const lines = content.replace(/\r\n/g, "\n").replace(/\r/g, "\n").split("\n");
  const output: string[] = [];
  const cscoEvidenceTitlePattern = /^【CSCO\s*RAG\s*检索证据】$/;
  const cscoCitationPattern = /^【CSCO-\d+\s*\|[^】]+】$/;
  const bracketSectionPattern = /^【(?!CSCO-\d+\s*\|).+】$/;
  const markdownHeadingPattern = /^#{1,6}\s+/;
  const textHeadingPattern = /^(治疗方案|推荐方案|用药建议|参考文献|PubMed|证据来源|注意事项|随访|结论|分析流程|Summary|Recommendation|References)[：:]?/i;
  let inCscoEvidence = false;

  lines.forEach((line) => {
    const trimmed = line.trim();
    const isEvidenceTitle = cscoEvidenceTitlePattern.test(trimmed);
    const isCscoCitation = cscoCitationPattern.test(trimmed);
    const startsNextSection =
      Boolean(trimmed) &&
      !isCscoCitation &&
      (bracketSectionPattern.test(trimmed) || markdownHeadingPattern.test(trimmed) || textHeadingPattern.test(trimmed));

    if (isEvidenceTitle || isCscoCitation) {
      inCscoEvidence = true;
      output.push(line);
      return;
    }

    if (inCscoEvidence) {
      if (startsNextSection) {
        inCscoEvidence = false;
        output.push(line);
      }
      return;
    }

    output.push(line);
  });

  return output.join("\n").replace(/\n{3,}/g, "\n\n").trim();
}

function formatMessageForBubble(content: string, visualizations: AnalysisVisualization[]) {
  const displayContent = filterPubmedEvidenceSections(
    collapseCscoEvidenceSnippets(filterCscoRagEvidenceSections(content))
  );
  if (!visualizations.length) return displayContent;

  const compact = displayContent
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n")
    .split("\n")
    .filter((line) => {
      const trimmed = line.trim();
      if (!trimmed) return true;
      if (extractPngPaths(trimmed).length) return false;
      if (/\bcsv\b/i.test(trimmed) && /(权重|分布|weight|distribution|attention)/i.test(trimmed)) return false;
      if (/\.csv\b/i.test(trimmed)) return false;
      if (/\.(npz|json)\b/i.test(trimmed)) return false;
      if (/(positive_raw_minmax|negative_inhibition_raw_minmax|positive_softmax|negative_inhibition_softmax|neutral_zero|neutral_uniform_softmax)/i.test(trimmed)) return false;
      if (trimmed.includes("计分模式") || trimmed.includes("负向抑制热力图说明") || trimmed.includes("抑制越强")) return false;
      if (/p_in_g_att|g_in_p_att|top_omic|最高频omic|相关.*omic|病理切片关注的基因组/i.test(trimmed)) return false;
      if (/transmil/i.test(trimmed) && /(attention|heatmap|热力图)/i.test(trimmed)) return false;
      if (/attention/i.test(trimmed) && /(summary|统计|原始|heatmap|热力图)/i.test(trimmed)) return false;
      return true;
    })
    .join("\n")
    .trim();

  return compact || displayContent;
}

function splitMarkdownRow(line: string) {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());
}

function isMarkdownTableSeparator(line: string) {
  const cells = splitMarkdownRow(line);
  return cells.length > 1 && cells.every((cell) => /^:?-{3,}:?$/.test(cell.replace(/\s+/g, "")));
}

function isMarkdownTableStart(lines: string[], index: number) {
  return lines[index]?.trim().startsWith("|") && isMarkdownTableSeparator(lines[index + 1] || "");
}

function normalizeTableCells(cells: string[], columnCount: number) {
  if (cells.length === columnCount) return cells;
  if (cells.length < columnCount) return [...cells, ...Array.from({ length: columnCount - cells.length }, () => "")];
  return [...cells.slice(0, columnCount - 1), cells.slice(columnCount - 1).join(" | ")];
}

function renderInlineMarkdown(text: string, keyPrefix: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  const pattern = /(\[[^\]]+\]\(https?:\/\/[^)\s]+\)|https?:\/\/[^\s<)]+|`[^`]+`|\*\*[^*]+\*\*)/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(text))) {
    if (match.index > lastIndex) {
      nodes.push(text.slice(lastIndex, match.index));
    }

    const token = match[0];
    const nodeKey = `${keyPrefix}-${match.index}`;
    const linkMatch = token.match(/^\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)$/);

    if (linkMatch) {
      nodes.push(
        <a key={nodeKey} href={linkMatch[2]} target="_blank" rel="noreferrer">
          {linkMatch[1]}
        </a>
      );
    } else if (/^https?:\/\//.test(token)) {
      nodes.push(
        <a key={nodeKey} href={token} target="_blank" rel="noreferrer">
          {token}
        </a>
      );
    } else if (token.startsWith("`")) {
      nodes.push(
        <code key={nodeKey} className="markdown-inline-code">
          {token.slice(1, -1)}
        </code>
      );
    } else if (token.startsWith("**")) {
      nodes.push(<strong key={nodeKey}>{renderInlineMarkdown(token.slice(2, -2), `${nodeKey}-strong`)}</strong>);
    }

    lastIndex = pattern.lastIndex;
  }

  if (lastIndex < text.length) {
    nodes.push(text.slice(lastIndex));
  }

  return nodes;
}

function renderInlineLines(lines: string[], keyPrefix: string): ReactNode[] {
  return lines.flatMap((line, index) => {
    const nodes = renderInlineMarkdown(line, `${keyPrefix}-${index}`);
    return index === lines.length - 1 ? nodes : [...nodes, <br key={`${keyPrefix}-br-${index}`} />];
  });
}

function renderMarkdownBlocks(content: string): ReactNode[] {
  const lines = content.replace(/\r\n/g, "\n").replace(/\r/g, "\n").split("\n");
  const blocks: ReactNode[] = [];
  let index = 0;
  let blockIndex = 0;

  while (index < lines.length) {
    const line = lines[index];
    const trimmed = line.trim();

    if (!trimmed) {
      index += 1;
      continue;
    }

    if (trimmed.startsWith("```")) {
      const codeLines: string[] = [];
      index += 1;
      while (index < lines.length && !lines[index].trim().startsWith("```")) {
        codeLines.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) index += 1;
      blocks.push(
        <pre key={`code-${blockIndex}`} className="markdown-code-block">
          <code>{codeLines.join("\n")}</code>
        </pre>
      );
      blockIndex += 1;
      continue;
    }

    if (isMarkdownTableStart(lines, index)) {
      const headers = splitMarkdownRow(lines[index]);
      const rows: string[][] = [];
      index += 2;
      while (index < lines.length && lines[index].trim().startsWith("|")) {
        if (!isMarkdownTableSeparator(lines[index])) {
          rows.push(normalizeTableCells(splitMarkdownRow(lines[index]), headers.length));
        }
        index += 1;
      }
      blocks.push(
        <div key={`table-${blockIndex}`} className="markdown-table-wrap">
          <table>
            <thead>
              <tr>
                {headers.map((header, cellIndex) => (
                  <th key={`h-${cellIndex}`}>{renderInlineMarkdown(header, `table-${blockIndex}-h-${cellIndex}`)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, rowIndex) => (
                <tr key={`r-${rowIndex}`}>
                  {row.map((cell, cellIndex) => (
                    <td key={`c-${cellIndex}`}>
                      {renderInlineMarkdown(cell, `table-${blockIndex}-r-${rowIndex}-c-${cellIndex}`)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
      blockIndex += 1;
      continue;
    }

    const headingMatch = trimmed.match(/^(#{1,6})\s+(.+)$/);
    if (headingMatch) {
      const level = Math.min(headingMatch[1].length, 6);
      const HeadingTag = `h${level}` as keyof JSX.IntrinsicElements;
      blocks.push(
        <HeadingTag key={`heading-${blockIndex}`}>
          {renderInlineMarkdown(headingMatch[2], `heading-${blockIndex}`)}
        </HeadingTag>
      );
      blockIndex += 1;
      index += 1;
      continue;
    }

    if (/^\s*[-*+]\s+/.test(line)) {
      const items: string[] = [];
      while (index < lines.length && /^\s*[-*+]\s+/.test(lines[index])) {
        items.push(lines[index].replace(/^\s*[-*+]\s+/, "").trim());
        index += 1;
      }
      blocks.push(
        <ul key={`ul-${blockIndex}`}>
          {items.map((item, itemIndex) => (
            <li key={itemIndex}>{renderInlineMarkdown(item, `ul-${blockIndex}-${itemIndex}`)}</li>
          ))}
        </ul>
      );
      blockIndex += 1;
      continue;
    }

    if (/^\s*\d+[.)]\s+/.test(line)) {
      const items: string[] = [];
      while (index < lines.length && /^\s*\d+[.)]\s+/.test(lines[index])) {
        items.push(lines[index].replace(/^\s*\d+[.)]\s+/, "").trim());
        index += 1;
      }
      blocks.push(
        <ol key={`ol-${blockIndex}`}>
          {items.map((item, itemIndex) => (
            <li key={itemIndex}>{renderInlineMarkdown(item, `ol-${blockIndex}-${itemIndex}`)}</li>
          ))}
        </ol>
      );
      blockIndex += 1;
      continue;
    }

    const paragraphLines: string[] = [];
    while (index < lines.length) {
      const current = lines[index];
      const currentTrimmed = current.trim();
      if (!currentTrimmed) break;
      if (
        paragraphLines.length &&
        (currentTrimmed.startsWith("```") ||
          isMarkdownTableStart(lines, index) ||
          /^#{1,6}\s+/.test(currentTrimmed) ||
          /^\s*[-*+]\s+/.test(current) ||
          /^\s*\d+[.)]\s+/.test(current))
      ) {
        break;
      }
      paragraphLines.push(currentTrimmed);
      index += 1;
    }

    blocks.push(<p key={`p-${blockIndex}`}>{renderInlineLines(paragraphLines, `p-${blockIndex}`)}</p>);
    blockIndex += 1;
  }

  return blocks;
}

function hasMarkdownTable(content: string) {
  const lines = content.replace(/\r\n/g, "\n").replace(/\r/g, "\n").split("\n");
  return lines.some((_, index) => isMarkdownTableStart(lines, index));
}

function MarkdownContent({ content, className = "" }: { content: string; className?: string }) {
  const classes = ["markdown-content", className].filter(Boolean).join(" ");
  return <div className={classes}>{renderMarkdownBlocks(content)}</div>;
}

async function readJson<T>(response: Response): Promise<T> {
  const text = await response.text();
  let payload: { detail?: unknown } = {};
  try {
    payload = text ? (JSON.parse(text) as { detail?: unknown }) : {};
  } catch {
    payload = { detail: text };
  }

  if (!response.ok) {
    throw new Error(typeof payload.detail === "string" ? payload.detail : text || `HTTP ${response.status}`);
  }
  return payload as T;
}

function postUpload<T>(
  path: string,
  file: File,
  sessionId: string | null,
  onProgress?: (value: number) => void
): Promise<T> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("file", file);
    if (sessionId) form.append("session_id", sessionId);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", apiUrl(path), true);
    xhr.responseType = "text";

    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable || !onProgress) return;
      const percent = Math.round((event.loaded / event.total) * 100);
      onProgress(Math.min(99, Math.max(1, percent)));
    };

    xhr.onload = () => {
      let payload: unknown = {};
      try {
        payload = xhr.responseText ? JSON.parse(xhr.responseText) : {};
      } catch {
        payload = { detail: xhr.responseText };
      }

      if (xhr.status < 200 || xhr.status >= 300) {
        const detail =
          typeof payload === "object" && payload !== null && "detail" in payload
            ? (payload as { detail?: unknown }).detail
            : "";
        reject(new Error(typeof detail === "string" ? detail : xhr.responseText || `HTTP ${xhr.status}`));
        return;
      }

      resolve(payload as T);
    };

    xhr.onerror = () => reject(new Error("网络错误，上传失败"));
    xhr.onabort = () => reject(new Error("上传已取消"));
    xhr.ontimeout = () => reject(new Error("上传超时"));

    onProgress?.(1);
    xhr.send(form);
  });
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(apiUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  return readJson<T>(response);
}

async function copyTextToClipboard(text: string) {
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return;
    } catch {
      // Fall back to a hidden textarea when clipboard permission is denied.
    }
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "true");
  textarea.style.position = "fixed";
  textarea.style.left = "-9999px";
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand("copy");
  document.body.removeChild(textarea);
}

async function createRemoteSession() {
  const response = await fetch(apiUrl("/api/session"), { method: "POST" });
  const data = await readJson<{ session_id: string }>(response);
  return data.session_id;
}

function startSyntheticProgress(setter: (value: number) => void, ceiling = 90, intervalMs = 600) {
  let value = 6;
  setter(value);
  const timer = window.setInterval(() => {
    const step = Math.max(1, Math.round((ceiling - value) * 0.12));
    value = Math.min(ceiling, value + step);
    setter(value);
  }, intervalMs);

  return () => window.clearInterval(timer);
}

function loadStoredCases(): CaseRecord[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(CASE_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as CaseRecord[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveStoredCases(cases: CaseRecord[]) {
  try {
    window.localStorage.setItem(CASE_STORAGE_KEY, JSON.stringify(cases.slice(0, 40)));
  } catch {
    // localStorage may be unavailable in private browsing or locked-down environments.
  }
}

const initialStoredCases = loadStoredCases();
const initialCurrentCaseId = initialStoredCases[0]?.id || createLocalCaseId();
const initialCurrentCase = initialStoredCases[0];

function normalizeRestoredStage(record?: CaseRecord): WorkStage {
  if (!record) return "idle";
  if (
    record.stage === "agent-running" ||
    record.stage === "clam-running" ||
    record.stage === "uploading-pathology" ||
    record.stage === "uploading-genome"
  ) {
    if (record.pathologyPt && record.genomeFile) return "genome-ready";
    if (record.pathologyPt) return "pathology-ready";
    if (record.genomeFile) return "genome-ready";
    return "idle";
  }
  return record.stage;
}

function modelLabel(model: string) {
  return llmModelOptions.find((item) => item.value === model)?.label || model;
}

function AnalysisFlowArrow() {
  return <div className="analysis-flow-arrow" aria-hidden="true" />;
}

function AnalysisFlowActionButton({
  cta,
  payload,
  onOpenLiterature,
  onOpenCscoRag
}: {
  cta: AnalysisFlowNodeCta;
  payload?: string;
  onOpenLiterature?: (payload?: string) => void;
  onOpenCscoRag?: (payload?: string) => void;
}) {
  const isLiterature = cta === "literature";
  const Icon = isLiterature ? FileText : BookOpenCheck;
  const label = isLiterature ? "检索文献" : "CSCO RAG";
  const onClick = isLiterature ? onOpenLiterature : onOpenCscoRag;

  return (
    <button
      type="button"
      className="analysis-flow-context-btn"
      title={isLiterature ? "查看本病例检索到的 PubMed 文献" : "查看本病例 RAG 检索到的 CSCO 指南片段"}
      onClick={() => onClick?.(payload)}
      disabled={!onClick}
    >
      <Icon size={13} />
      <span>{label}</span>
    </button>
  );
}

function AnalysisFlowTextWithEmbeddedAction({
  text,
  markdown,
  cta,
  ctaPayload,
  onOpenLiterature,
  onOpenCscoRag
}: {
  text: string;
  markdown: boolean;
  cta: AnalysisFlowNodeCta;
  ctaPayload?: string;
  onOpenLiterature?: (payload?: string) => void;
  onOpenCscoRag?: (payload?: string) => void;
}) {
  const lines = normalizeAnalysisText(text).split("\n");
  const queryIndex = lines.findIndex((line) => /CSCO\s*RAG检索查询\s*[:：]/.test(line));
  const beforeText = displayAnalysisText(lines.slice(0, queryIndex >= 0 ? queryIndex + 1 : 1).join("\n"));
  const afterText = displayAnalysisText(lines.slice(queryIndex >= 0 ? queryIndex + 1 : 1).join("\n"));

  return (
    <div className="analysis-flow-action-content">
      {beforeText ? <span className="analysis-flow-action-text">{beforeText}</span> : null}
      <div className="analysis-flow-embedded-action">
        <AnalysisFlowActionButton
          cta={cta}
          payload={ctaPayload}
          onOpenLiterature={onOpenLiterature}
          onOpenCscoRag={onOpenCscoRag}
        />
      </div>
      {afterText ? (
        markdown ? (
          <MarkdownContent content={afterText} className="analysis-flow-markdown" />
        ) : (
          <span className="analysis-flow-action-text">{afterText}</span>
        )
      ) : null}
    </div>
  );
}

function AnalysisFlowBox({
  label,
  text,
  kind,
  title,
  markdown = false,
  cta,
  ctaPlacement,
  ctaPayload,
  onOpenLiterature,
  onOpenCscoRag
}: {
  label: string;
  text: string;
  kind: "task" | "final" | AnalysisFlowNodeKind;
  title?: string;
  markdown?: boolean;
  cta?: AnalysisFlowNodeCta;
  ctaPlacement?: AnalysisFlowNodeCtaPlacement;
  ctaPayload?: string;
  onOpenLiterature?: (payload?: string) => void;
  onOpenCscoRag?: (payload?: string) => void;
}) {
  const titleAction = cta && ctaPlacement === "title";
  const embeddedAction = cta && ctaPlacement === "after-csco-query";

  return (
    <div className={`analysis-flow-box ${kind} ${markdown ? "has-markdown" : ""}`} title={title || text}>
      <div className={`analysis-flow-box-header ${titleAction ? "has-action" : ""}`}>
        <strong>{label}</strong>
        {titleAction ? (
          <AnalysisFlowActionButton
            cta={cta}
            payload={ctaPayload}
            onOpenLiterature={onOpenLiterature}
            onOpenCscoRag={onOpenCscoRag}
          />
        ) : null}
      </div>
      {embeddedAction ? (
        <AnalysisFlowTextWithEmbeddedAction
          text={text}
          markdown={markdown}
          cta={cta}
          ctaPayload={ctaPayload}
          onOpenLiterature={onOpenLiterature}
          onOpenCscoRag={onOpenCscoRag}
        />
      ) : markdown ? (
        <MarkdownContent content={text} className="analysis-flow-markdown" />
      ) : (
        <span>{text}</span>
      )}
    </div>
  );
}

function AnalysisFlowChart({
  log,
  userTask,
  finalAnswer,
  onOpenLiterature,
  onOpenCscoRag
}: {
  log: string;
  userTask: string;
  finalAnswer: string;
  onOpenLiterature?: (payload?: string) => void;
  onOpenCscoRag?: (payload?: string) => void;
}) {
  const flow = useMemo(() => parseAnalysisFlow(log, userTask, finalAnswer), [log, userTask, finalAnswer]);

  if (!log.trim()) {
    return <div className="analysis-flow-empty">暂无分析流程输出</div>;
  }

  return (
    <div className="analysis-flow-diagram" aria-label="医学任务分析流程图">
      <div className="analysis-flow-body">
        <AnalysisFlowBox label="用户输入任务:" text={flow.taskText} kind="task" title={userTask || flow.taskText} />

        {flow.rounds.length ? (
          flow.rounds.map((round, roundIndex) => (
            <div className="analysis-flow-round" key={round.id}>
              <div className="analysis-round-label">
                <span>{formatRoundLabel(round.roundNumber ?? roundIndex + 1)}</span>
              </div>
              <div className="analysis-round-brace" aria-hidden="true" />
              <div className="analysis-round-steps">
                {round.nodes.map((node) => (
                  <div className="analysis-flow-step" key={node.id}>
                    <AnalysisFlowArrow />
                    <AnalysisFlowBox
                      label={node.label}
                      text={node.text}
                      kind={node.kind}
                      title={`${node.label} ${node.fullText}`}
                      markdown={node.markdown}
                      cta={node.cta}
                      ctaPlacement={node.ctaPlacement}
                      ctaPayload={node.ctaPayload}
                      onOpenLiterature={onOpenLiterature}
                      onOpenCscoRag={onOpenCscoRag}
                    />
                  </div>
                ))}
              </div>
            </div>
          ))
        ) : (
          <div className="analysis-flow-step">
            <AnalysisFlowArrow />
            <AnalysisFlowBox label="分析日志:" text={flow.fallbackText} kind="tool-flow" title={log} />
          </div>
        )}

        {flow.finalText ? (
          <div className="analysis-flow-step analysis-flow-final-step">
            <AnalysisFlowArrow />
            <AnalysisFlowBox
              label="Final answer:"
              text={flow.finalText}
              kind="final"
              title={finalAnswer || flow.finalText}
              markdown={flow.finalMarkdown}
              onOpenLiterature={onOpenLiterature}
              onOpenCscoRag={onOpenCscoRag}
            />
          </div>
        ) : null}
      </div>
    </div>
  );
}

function ChatBubble({
  message,
  visualizations,
  active,
  onCopy,
  onRetry,
  onOpenVisualizations,
  retryDisabled
}: {
  message: ChatMessage;
  visualizations: AnalysisVisualization[];
  active: boolean;
  onCopy: (content: string) => void;
  onRetry: () => void;
  onOpenVisualizations: () => void;
  retryDisabled: boolean;
}) {
  const displayContent = formatMessageForBubble(message.content, visualizations);
  const rendersMarkdown = message.role === "assistant";
  const tableClass = rendersMarkdown && hasMarkdownTable(displayContent) ? "has-markdown-table" : "";
  const literatureMarkers = rendersMarkdown ? getReferencedLiteratureMarkers(message.content) : null;
  const referencedCscoEvidence = rendersMarkdown ? filterReferencedCscoCitations(message.content) : [];
  const hasEvidence =
    rendersMarkdown &&
    (referencedCscoEvidence.length > 0 ||
      Boolean(literatureMarkers && (literatureMarkers.pubmedIds.size || literatureMarkers.pmids.size)));
  const hasDetails = Boolean(visualizations.length || message.analysisLog?.trim() || hasEvidence);

  return (
    <div
      className={`message-row ${message.role} ${hasDetails ? "has-visualizations" : ""} ${tableClass}`}
      data-message-id={message.id}
    >
      <div className="message-content-wrap">
        <div className="message-bubble">
          {rendersMarkdown ? <MarkdownContent content={displayContent} /> : <pre>{displayContent}</pre>}
        </div>
        {rendersMarkdown ? (
          <div className="message-actions" aria-label="智能体回复操作">
            <button
              type="button"
              className="message-action-btn"
              aria-label="复制"
              onClick={() => {
                void onCopy(displayContent);
              }}
            >
              <Copy size={14} />
              <span>复制</span>
            </button>
            <button
              type="button"
              className="message-action-btn"
              aria-label="重试"
              disabled={retryDisabled}
              onClick={onRetry}
            >
              <RefreshCw size={14} />
              <span>重试</span>
            </button>
            {hasDetails ? (
              <button
                type="button"
                className={`message-action-btn message-action-show ${active ? "active" : ""}`}
                aria-label="展示"
                aria-expanded={active}
                aria-pressed={active}
                onClick={onOpenVisualizations}
              >
                <ChevronRight size={15} />
              </button>
            ) : null}
          </div>
        ) : null}
      </div>
    </div>
  );
}

function CscoEvidencePreview({ citation, onOpen }: { citation: CscoCitation; onOpen: (citation: CscoCitation) => void }) {
  const [imageFailed, setImageFailed] = useState(false);
  const showImage = Boolean(citation.imageUrl && !imageFailed);

  return (
    <article className="csco-evidence-item">
      <div className="csco-evidence-head">
        <strong>CSCO-{citation.id}</strong>
        <span title={citation.filename}>
          {citation.filename} · {citation.pageLabel}
        </span>
      </div>
      <div className="csco-pdf-preview">
        {showImage ? (
          <button
            type="button"
            className="csco-pdf-preview-trigger"
            aria-label={`查看 ${citation.filename} ${citation.pageLabel} 预览图`}
            title="点击查看预览图"
            onClick={() => onOpen(citation)}
          >
            <img
              src={citation.imageUrl}
              alt={`${citation.filename} ${citation.pageLabel}`}
              draggable={false}
              onError={() => setImageFailed(true)}
            />
            <span className="heatmap-zoom-icon csco-zoom-icon" aria-hidden="true">
              <ZoomIn size={14} />
            </span>
          </button>
        ) : (
          <iframe title={`${citation.filename} ${citation.pageLabel}`} src={citation.pdfUrl} />
        )}
      </div>
    </article>
  );
}

function RecentCaseItem({
  record,
  active,
  menuOpen,
  onSelect,
  onToggleMenu,
  onRename,
  onDelete
}: {
  record: CaseRecord;
  active: boolean;
  menuOpen: boolean;
  onSelect: (record: CaseRecord) => void;
  onToggleMenu: (recordId: string) => void;
  onRename: (record: CaseRecord) => void;
  onDelete: (record: CaseRecord) => void;
}) {
  const latest =
    record.latestOutput ||
    [...record.messages].reverse().find((message) => message.role === "user")?.content ||
    "尚未开始分析";

  return (
    <div className={`recent-case ${active ? "active" : ""}`}>
      <button type="button" className="recent-main" onClick={() => onSelect(record)}>
        <span>
          <strong>{record.title}</strong>
          <small>{latest}</small>
        </span>
      </button>
      <button type="button" className="case-menu-trigger" aria-label="病例操作" onClick={() => onToggleMenu(record.id)}>
        {menuOpen ? <MoreHorizontal size={15} /> : <ChevronRight size={15} />}
      </button>
      {menuOpen ? (
        <div className="case-menu">
          <button type="button" onClick={() => onRename(record)}>
            <Pencil size={14} />
            重命名病例
          </button>
          <button type="button" className="danger" onClick={() => onDelete(record)}>
            <Trash2 size={14} />
            删除病例
          </button>
        </div>
      ) : null}
    </div>
  );
}

export default function App() {
  const [currentCaseId, setCurrentCaseId] = useState(initialCurrentCaseId);
  const [cases, setCases] = useState<CaseRecord[]>(initialStoredCases);
  const [caseTitleOverride, setCaseTitleOverride] = useState(initialCurrentCase?.customTitle || "");
  const [searchQuery, setSearchQuery] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(initialCurrentCase?.sessionId || null);
  const [cancerType, setCancerType] = useState(initialCurrentCase?.cancerType || "BRCA");
  const [llmModel, setLlmModel] = useState(initialCurrentCase?.llmModel || DEFAULT_LLM_MODEL);
  const [stage, setStage] = useState<WorkStage>(normalizeRestoredStage(initialCurrentCase));
  const [messages, setMessages] = useState<ChatMessage[]>(initialCurrentCase?.messages || []);
  const [wsiFile, setWsiFile] = useState<UploadedFileMeta | null>(initialCurrentCase?.wsiFile || null);
  const [genomeFile, setGenomeFile] = useState<UploadedFileMeta | null>(initialCurrentCase?.genomeFile || null);
  const [genomeColumns, setGenomeColumns] = useState<string[]>(initialCurrentCase?.genomeColumns || []);
  const [patientSummary, setPatientSummary] = useState<PatientSummary>(initialCurrentCase?.patientSummary || {});
  const [pathologyJpgUrl, setPathologyJpgUrl] = useState(initialCurrentCase?.pathologyJpgUrl || "");
  const [pathologyPt, setPathologyPt] = useState(initialCurrentCase?.pathologyPt || "");
  const [query, setQuery] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [uploadingWsi, setUploadingWsi] = useState(false);
  const [uploadingGenome, setUploadingGenome] = useState(false);
  const [pathologyProgress, setPathologyProgress] = useState(pathologyPt ? 100 : 0);
  const [genomeProgress, setGenomeProgress] = useState(genomeFile ? 100 : 0);
  const [clamProgress, setClamProgress] = useState(pathologyPt ? 100 : 0);
  const [rightWidth, setRightWidth] = useState(360);
  const [leftCollapsed, setLeftCollapsed] = useState(false);
  const [rightCollapsed, setRightCollapsed] = useState(false);
  const [uploadMenuOpen, setUploadMenuOpen] = useState(false);
  const [modelMenuOpen, setModelMenuOpen] = useState(false);
  const [activeCaseMenuId, setActiveCaseMenuId] = useState<string | null>(null);
  const [knowledgeOpen, setKnowledgeOpen] = useState(false);
  const [knowledgeLoading, setKnowledgeLoading] = useState(false);
  const [knowledgeUploading, setKnowledgeUploading] = useState(false);
  const [knowledgeDeletingName, setKnowledgeDeletingName] = useState<string | null>(null);
  const [knowledgeFiles, setKnowledgeFiles] = useState<KnowledgeFile[]>([]);
  const [literatureOpen, setLiteratureOpen] = useState(false);
  const [literatureLoading, setLiteratureLoading] = useState(false);
  const [literatureText, setLiteratureText] = useState(initialCurrentCase?.literatureText || "");
  const [contextLiteratureText, setContextLiteratureText] = useState("");
  const [cscoRagOpen, setCscoRagOpen] = useState(false);
  const [activeVisualizationMessageId, setActiveVisualizationMessageId] = useState<number | null>(null);
  const [expandedSplitCard, setExpandedSplitCard] = useState<ExpandedSplitCard>(null);
  const [previewHeatmap, setPreviewHeatmap] = useState<HeatmapItem | null>(null);
  const [previewCscoCitation, setPreviewCscoCitation] = useState<CscoCitation | null>(null);
  const [pathologyPreviewOpen, setPathologyPreviewOpen] = useState(false);

  const pathologyInputRef = useRef<HTMLInputElement>(null);
  const genomeInputRef = useRef<HTMLInputElement>(null);
  const knowledgeInputRef = useRef<HTMLInputElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const appShellRef = useRef<HTMLDivElement>(null);
  const sidebarRef = useRef<HTMLElement>(null);
  const conversationRef = useRef<HTMLDivElement>(null);
  const userMessageScrollTargetRef = useRef<number | null>(null);
  const rightWidthRef = useRef(rightWidth);
  const resizeFrameRef = useRef<number | null>(null);

  const genomeFileRef = useRef(genomeFile);
  useEffect(() => {
    genomeFileRef.current = genomeFile;
  }, [genomeFile]);

  useEffect(() => {
    rightWidthRef.current = rightWidth;
  }, [rightWidth]);

  useEffect(() => {
    if (searchOpen) {
      window.requestAnimationFrame(() => {
        if (sidebarRef.current) {
          sidebarRef.current.scrollLeft = 0;
        }
        searchInputRef.current?.focus({ preventScroll: true });
        if (sidebarRef.current) {
          sidebarRef.current.scrollLeft = 0;
        }
      });
    }
  }, [searchOpen]);

  useEffect(() => {
    const targetId = userMessageScrollTargetRef.current;
    const conversation = conversationRef.current;
    if (!targetId || !conversation) return;

    const target = conversation.querySelector<HTMLElement>(`[data-message-id="${targetId}"]`);
    if (!target) return;

    const animationFrame = window.requestAnimationFrame(() => {
      target.scrollIntoView({ behavior: "smooth", block: "start", inline: "nearest" });
      userMessageScrollTargetRef.current = null;
    });

    return () => window.cancelAnimationFrame(animationFrame);
  }, [messages]);

  const hasClamFeature = Boolean(pathologyPt);
  const agentBusy = stage === "agent-running";
  const clamBusy = stage === "clam-running";
  const latestAssistantMessage = [...messages].reverse().find((message) => message.role === "assistant") || null;
  const latestAssistant = latestAssistantMessage?.content || "";
  const visualizationsByMessageId = useMemo(() => {
    const parsed = new Map<number, AnalysisVisualization[]>();
    messages.forEach((message) => {
      if (message.role !== "assistant") return;
      const visualizations = parseAnalysisVisualizations(message.content, sessionId);
      if (visualizations.length) parsed.set(message.id, visualizations);
    });
    return parsed;
  }, [messages, sessionId]);
  const activeDetailMessage =
    activeVisualizationMessageId === null
      ? null
      : messages.find((message) => message.id === activeVisualizationMessageId) || null;
  const activeUserTask = useMemo(() => {
    if (!activeDetailMessage) return "";
    const messageIndex = messages.findIndex((message) => message.id === activeDetailMessage.id);
    if (messageIndex < 0) return "";
    return [...messages.slice(0, messageIndex)].reverse().find((message) => message.role === "user")?.content || "";
  }, [activeDetailMessage, messages]);
  const activeVisualizations =
    activeVisualizationMessageId === null ? [] : visualizationsByMessageId.get(activeVisualizationMessageId) || [];
  const activeAnalysisLog = activeDetailMessage?.analysisLog?.trim() || "";
  const literatureItems = useMemo(() => parseLiteratureItems(literatureText), [literatureText]);
  const panelLiteratureText = contextLiteratureText || literatureText;
  const panelLiteratureItems = useMemo(() => parseLiteratureItems(panelLiteratureText), [panelLiteratureText]);
  const activeAlignedLiteratureItems = useMemo(() => {
    if (!activeDetailMessage) return literatureItems;
    const pubmedToolText = extractLatestPubmedToolText(activeAnalysisLog);
    if (!pubmedToolText) return literatureItems;
    const evidenceText = extractPubmedEvidenceText(activeDetailMessage.content) || literatureText;
    const alignedText = alignPubmedEvidenceToToolReferences(pubmedToolText, evidenceText);
    const alignedItems = parseLiteratureItems(alignedText);
    return alignedItems.length ? alignedItems : literatureItems;
  }, [activeAnalysisLog, activeDetailMessage, literatureItems, literatureText]);
  const activeCscoRagEvidenceText = useMemo(() => {
    if (activeDetailMessage) {
      return extractCscoRagEvidenceText(activeDetailMessage.content);
    }
    for (const message of [...messages].reverse()) {
      if (message.role !== "assistant") continue;
      const evidenceText = extractCscoRagEvidenceText(message.content);
      if (evidenceText) return evidenceText;
    }
    return "";
  }, [activeDetailMessage, messages]);
  const cscoRagEvidenceItems = useMemo(() => parseCscoRagEvidenceItems(activeCscoRagEvidenceText), [activeCscoRagEvidenceText]);
  const activeReferencedCscoCitations = useMemo(
    () => (activeDetailMessage ? filterReferencedCscoCitations(activeDetailMessage.content) : []),
    [activeDetailMessage]
  );
  const activeLiteratureEvidence = useMemo(
    () => (activeDetailMessage ? filterReferencedLiteratureItems(activeAlignedLiteratureItems, activeDetailMessage.content) : []),
    [activeAlignedLiteratureItems, activeDetailMessage]
  );
  const activeMessageLooksLikeTreatment = Boolean(
    activeDetailMessage && /(治疗|方案|CSCO|PubMed|PMID|文献证据|用药|免疫|化疗|靶向)/i.test(activeDetailMessage.content)
  );
  const hasActiveEvidence = Boolean(activeReferencedCscoCitations.length || activeLiteratureEvidence.length);
  const cmtaVisualization = activeVisualizations.find((item) => item.type === "cmta");
  const otherVisualizations = activeVisualizations.filter((item) => item.type !== "cmta");
  const rightPanelMode = knowledgeOpen
    ? "knowledge"
    : literatureOpen
      ? "literature"
      : cscoRagOpen
        ? "csco-rag"
        : activeDetailMessage && (activeVisualizations.length || activeAnalysisLog || hasActiveEvidence)
          ? "visualizations"
          : null;
  const showDefaultRightContent = rightPanelMode === null;
  const showRightContent =
    Boolean(rightPanelMode) ||
    Boolean(errorMessage) ||
    Object.keys(patientSummary).length > 0 ||
    Boolean(pathologyJpgUrl) ||
    Boolean(pathologyPt) ||
    Boolean(wsiFile) ||
    Boolean(genomeFile) ||
    stage === "clam-running";

  useEffect(() => {
    setPreviewHeatmap(null);
    setPreviewCscoCitation(null);
    setContextLiteratureText("");
    setExpandedSplitCard(null);
  }, [activeVisualizationMessageId, currentCaseId]);

  useEffect(() => {
    if (
      rightPanelMode !== "visualizations" ||
      !activeMessageLooksLikeTreatment ||
      literatureText ||
      literatureLoading ||
      !sessionId
    ) {
      return;
    }

    let cancelled = false;
    setLiteratureLoading(true);
    fetch(apiUrl(`/api/session/${sessionId}/literature`))
      .then((response) => readJson<LiteratureResponse>(response))
      .then((data) => {
        if (!cancelled) setLiteratureText(data.literature || "");
      })
      .catch((error) => {
        if (!cancelled) setErrorMessage(error instanceof Error ? error.message : "无法加载检索文献");
      })
      .finally(() => {
        if (!cancelled) setLiteratureLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [activeDetailMessage?.id, activeMessageLooksLikeTreatment, literatureLoading, literatureText, rightPanelMode, sessionId]);

  useEffect(() => {
    const snapshot: CaseRecord = {
      id: currentCaseId,
      title: caseTitleOverride || getCaseTitle(patientSummary, wsiFile, genomeFile),
      customTitle: caseTitleOverride || undefined,
      sessionId,
      cancerType,
      llmModel,
      updatedAt: Date.now(),
      stage,
      messages,
      patientSummary,
      wsiFile,
      genomeFile,
      genomeColumns,
      pathologyJpgUrl,
      pathologyPt,
      latestOutput: latestAssistant,
      literatureText
    };

    setCases((previous) => {
      const next = [snapshot, ...previous.filter((record) => record.id !== currentCaseId)]
        .sort((a, b) => b.updatedAt - a.updatedAt)
        .slice(0, 40);
      saveStoredCases(next);
      return next;
    });
  }, [
    caseTitleOverride,
    cancerType,
    currentCaseId,
    genomeColumns,
    genomeFile,
    latestAssistant,
    literatureText,
    llmModel,
    messages,
    patientSummary,
    pathologyJpgUrl,
    pathologyPt,
    sessionId,
    stage,
    wsiFile
  ]);

  const filteredCases = useMemo(() => {
    const keyword = searchQuery.trim().toLowerCase();
    const sorted = [...cases].sort((a, b) => b.updatedAt - a.updatedAt);
    if (!keyword) return sorted;
    return sorted.filter((record) => {
      const haystack = [
        record.title,
        record.cancerType,
        record.llmModel,
        record.latestOutput,
        record.sessionId || "",
        ...(record.genomeColumns || []),
        ...record.messages.map((message) => message.content)
      ]
        .join(" ")
        .toLowerCase();
      return haystack.includes(keyword);
    });
  }, [cases, searchQuery]);

  const startNewCase = () => {
    setCurrentCaseId(createLocalCaseId());
    setCaseTitleOverride("");
    setSessionId(null);
    setCancerType("BRCA");
    setLlmModel(DEFAULT_LLM_MODEL);
    setStage("idle");
    setMessages([]);
    setWsiFile(null);
    setGenomeFile(null);
    setGenomeColumns([]);
    setPatientSummary({});
    setPathologyJpgUrl("");
    setPathologyPt("");
    setQuery("");
    setErrorMessage("");
    setPathologyProgress(0);
    setGenomeProgress(0);
    setClamProgress(0);
    setKnowledgeOpen(false);
    setLiteratureOpen(false);
    setLiteratureText("");
    setContextLiteratureText("");
    setCscoRagOpen(false);
    setActiveVisualizationMessageId(null);
    setUploadMenuOpen(false);
    setModelMenuOpen(false);
    setPathologyPreviewOpen(false);
  };

  const loadCase = (record: CaseRecord) => {
    const restoredStage = normalizeRestoredStage(record);

    setCurrentCaseId(record.id);
    setCaseTitleOverride(record.customTitle || "");
    setSessionId(record.sessionId);
    setCancerType(record.cancerType);
    setLlmModel(record.llmModel || DEFAULT_LLM_MODEL);
    setStage(restoredStage);
    setMessages(record.messages || []);
    setWsiFile(record.wsiFile || null);
    setGenomeFile(record.genomeFile || null);
    setGenomeColumns(record.genomeColumns || []);
    setPatientSummary(record.patientSummary || {});
    setPathologyJpgUrl(record.pathologyJpgUrl || "");
    setPathologyPt(record.pathologyPt || "");
    setQuery("");
    setErrorMessage("");
    setPathologyProgress(record.pathologyPt ? 100 : 0);
    setGenomeProgress(record.genomeFile ? 100 : 0);
    setClamProgress(record.pathologyPt ? 100 : 0);
    setKnowledgeOpen(false);
    setLiteratureOpen(false);
    setLiteratureText(record.literatureText || "");
    setContextLiteratureText("");
    setCscoRagOpen(false);
    setActiveCaseMenuId(null);
    setActiveVisualizationMessageId(null);
    setUploadMenuOpen(false);
    setModelMenuOpen(false);
    setPathologyPreviewOpen(false);
  };

  const renameCase = (record: CaseRecord) => {
    const nextTitle = window.prompt("请输入新的病例名称", record.title)?.trim();
    if (!nextTitle) {
      setActiveCaseMenuId(null);
      return;
    }

    if (record.id === currentCaseId) {
      setCaseTitleOverride(nextTitle);
    }

    setCases((previous) => {
      const next = previous.map((item) =>
        item.id === record.id ? { ...item, title: nextTitle, customTitle: nextTitle, updatedAt: Date.now() } : item
      );
      saveStoredCases(next);
      return next;
    });
    setActiveCaseMenuId(null);
  };

  const deleteCase = (record: CaseRecord) => {
    if (!window.confirm(`确定删除病例“${record.title}”吗？`)) return;

    const remaining = cases.filter((item) => item.id !== record.id);
    saveStoredCases(remaining);
    setCases(remaining);
    setActiveCaseMenuId(null);

    if (record.id === currentCaseId) {
      const nextRecord = remaining[0];
      if (nextRecord) {
        loadCase(nextRecord);
      } else {
        startNewCase();
      }
    }
  };

  const ensureSession = async () => {
    if (sessionId) return sessionId;
    const newSessionId = await createRemoteSession();
    setSessionId(newSessionId);
    return newSessionId;
  };

  const loadLiteratureForCurrentSession = async () => {
    const hasStructuredLiterature = literatureItems.length > 0;
    if ((literatureText && hasStructuredLiterature) || literatureLoading || !sessionId) return;

    setLiteratureLoading(true);
    setErrorMessage("");
    try {
      const data = await fetch(apiUrl(`/api/session/${sessionId}/literature`)).then((response) =>
        readJson<LiteratureResponse>(response)
      );
      setLiteratureText(data.literature || "");
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "无法加载检索文献");
    } finally {
      setLiteratureLoading(false);
    }
  };

  const loadKnowledgeFiles = async () => {
    setKnowledgeLoading(true);
    setErrorMessage("");
    try {
      const data = await fetch(apiUrl("/api/knowledge/csco")).then((response) => readJson<KnowledgeFilesResponse>(response));
      setKnowledgeFiles(data.files || []);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "无法加载 CSCO 治疗指南");
    } finally {
      setKnowledgeLoading(false);
    }
  };

  const openKnowledgeBase = async () => {
    const nextOpen = !knowledgeOpen;
    setKnowledgeOpen(nextOpen);
    setLiteratureOpen(false);
    setCscoRagOpen(false);
    setContextLiteratureText("");
    setActiveVisualizationMessageId(null);
    if (nextOpen) setRightCollapsed(false);
    if (!nextOpen || knowledgeLoading) return;

    await loadKnowledgeFiles();
  };

  const handleKnowledgeUpload = async (file: File) => {
    setErrorMessage("");
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setErrorMessage("治疗指南目前仅支持上传 PDF 文件");
      return;
    }

    setKnowledgeUploading(true);
    try {
      const data = await postUpload<KnowledgeUploadResponse>("/api/knowledge/csco", file, null);
      setKnowledgeFiles(data.files || (data.file ? [data.file, ...knowledgeFiles] : knowledgeFiles));
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "治疗指南文件上传失败");
    } finally {
      setKnowledgeUploading(false);
    }
  };

  const deleteKnowledgeFile = async (file: KnowledgeFile) => {
    if (!window.confirm(`确定删除治疗指南文件“${file.name}”吗？`)) return;

    setKnowledgeDeletingName(file.name);
    setErrorMessage("");
    try {
      const data = await fetch(apiUrl(`/api/knowledge/csco/${encodeURIComponent(file.name)}`), {
        method: "DELETE"
      }).then((response) => readJson<KnowledgeFilesResponse>(response));
      setKnowledgeFiles(data.files || []);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "治疗指南文件删除失败");
    } finally {
      setKnowledgeDeletingName(null);
    }
  };

  const openAnalysisLiteraturePanel = (toolLiteratureText?: string) => {
    const toolText = displayAnalysisText(toolLiteratureText || "");
    const messageEvidenceText = extractPubmedEvidenceText(activeDetailMessage?.content || "");
    const exactToolLiterature = alignPubmedEvidenceToToolReferences(toolText, messageEvidenceText);
    setContextLiteratureText(exactToolLiterature);
    setLiteratureOpen(true);
    setKnowledgeOpen(false);
    setCscoRagOpen(false);
    setExpandedSplitCard(null);
    setRightCollapsed(false);
    if (!exactToolLiterature) {
      void loadLiteratureForCurrentSession();
    }
  };

  const openAnalysisCscoRagPanel = () => {
    setCscoRagOpen(true);
    setKnowledgeOpen(false);
    setLiteratureOpen(false);
    setContextLiteratureText("");
    setExpandedSplitCard(null);
    setRightCollapsed(false);
  };

  const returnToEvidenceAndFlowPanel = () => {
    setKnowledgeOpen(false);
    setLiteratureOpen(false);
    setCscoRagOpen(false);
    setContextLiteratureText("");
    setExpandedSplitCard(null);
    setRightCollapsed(false);
  };

  const runClamForSession = async (activeSessionId: string) => {
    setErrorMessage("");
    setStage("clam-running");
    const stopProgress = startSyntheticProgress(setClamProgress, 92, 1400);

    try {
      const data = await postJson<ClamRunResponse>("/api/clam/run", { session_id: activeSessionId });
      stopProgress();
      setClamProgress(100);
      setPathologyJpgUrl(absoluteApiUrl(data.pathology_jpg_url));
      setPathologyPt(data.pathology_pt || "");
      setStage(genomeFileRef.current ? "genome-ready" : "pathology-ready");
    } catch (error) {
      stopProgress();
      const message = error instanceof Error ? error.message : "CLAM 处理失败";
      setErrorMessage(message);
      setStage("failed");
    }
  };

  const handleWsiUpload = async (file: File) => {
    setUploadMenuOpen(false);
    setModelMenuOpen(false);
    setPathologyPreviewOpen(false);
    setErrorMessage("");
    setUploadingWsi(true);
    setStage("uploading-pathology");
    setPathologyJpgUrl("");
    setPathologyPt("");
    setClamProgress(0);
    setPathologyProgress(0);

    try {
      const data = await postUpload<UploadPathologyResponse>("/api/upload/pathology", file, sessionId, setPathologyProgress);
      setPathologyProgress(100);
      setSessionId(data.session_id);
      setWsiFile(fileToMeta(file));
      setUploadingWsi(false);
      await runClamForSession(data.session_id);
    } catch (error) {
      setUploadingWsi(false);
      setStage("failed");
      setErrorMessage(error instanceof Error ? error.message : "病理文件上传失败");
    }
  };

  const handleGenomeUpload = async (file: File) => {
    setUploadMenuOpen(false);
    setModelMenuOpen(false);
    setErrorMessage("");
    setUploadingGenome(true);
    setStage((previous) => (previous === "clam-running" ? previous : "uploading-genome"));
    setGenomeProgress(0);
    setGenomeColumns([]);

    try {
      const previewColumns = await readGenomeColumnPreview(file);
      const data = await postUpload<UploadGenomeResponse>("/api/upload/genome", file, sessionId, setGenomeProgress);
      setGenomeProgress(100);
      setSessionId(data.session_id);
      setGenomeFile(fileToMeta(file));
      setGenomeColumns(previewColumns);
      setPatientSummary(data.patient_summary || {});
      if (data.cancer_type) setCancerType(data.cancer_type);
      setStage((previous) => (previous === "clam-running" ? previous : "genome-ready"));
    } catch (error) {
      setStage("failed");
      setErrorMessage(error instanceof Error ? error.message : "基因组文件上传失败");
    } finally {
      setUploadingGenome(false);
    }
  };

  const runAgentQuery = async (rawQuery: string, retryMessageId?: number) => {
    const trimmed = rawQuery.trim();
    if (!trimmed || agentBusy) return;

    setErrorMessage("");
    setActiveVisualizationMessageId(null);
    setCscoRagOpen(false);
    setUploadMenuOpen(false);
    setModelMenuOpen(false);

    let activeSessionId: string;
    try {
      activeSessionId = await ensureSession();
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "无法创建病例会话");
      return;
    }

    const pendingId = retryMessageId ?? nextMessageId();
    if (retryMessageId) {
      setMessages((previous) =>
        previous.map((message) =>
          message.id === retryMessageId
            ? { ...message, content: "主智能体正在重新生成回复...", analysisLog: "" }
            : message
        )
      );
    } else {
      const userMessageId = nextMessageId();
      userMessageScrollTargetRef.current = userMessageId;
      setMessages((previous) => [
        ...previous,
        { id: userMessageId, role: "user", content: trimmed },
        { id: pendingId, role: "assistant", content: "主智能体正在分析问题并选择合适能力..." }
      ]);
      setQuery("");
    }
    setStage("agent-running");

    try {
      const data = await postJson<ChatResponse>("/api/agent/chat", {
        session_id: activeSessionId,
        query: trimmed,
        cancer_type: cancerType,
        llm_model: llmModel
      });
      if (data.cancer_type) setCancerType(data.cancer_type);
      if (data.llm_model) setLlmModel(data.llm_model);
      if (data.literature) setLiteratureText(data.literature);
      setMessages((previous) =>
        previous.map((message) =>
          message.id === pendingId ? { ...message, content: data.result, analysisLog: data.analysis_log || "" } : message
        )
      );
      setStage("completed");
    } catch (error) {
      const message = error instanceof Error ? error.message : "智能体调用失败";
      setMessages((previous) =>
        previous.map((item) =>
          item.id === pendingId ? { ...item, content: `调用失败：${message}`, analysisLog: "" } : item
        )
      );
      setErrorMessage(message);
      setStage("failed");
    }
  };

  const submitQuery = () => {
    void runAgentQuery(query);
  };

  const retryAssistantMessage = (messageId: number) => {
    const messageIndex = messages.findIndex((message) => message.id === messageId);
    if (messageIndex < 0) return;

    const previousUser = [...messages.slice(0, messageIndex)].reverse().find((message) => message.role === "user");
    if (!previousUser) return;
    void runAgentQuery(previousUser.content, messageId);
  };

  const pathologyStatus = hasClamFeature
    ? "病理数据已完成 CLAM 处理"
    : clamBusy
      ? `CLAM 处理中 ${clamProgress}%`
      : uploadingWsi
        ? `上传中 ${pathologyProgress}%`
        : "上传 WSI 后将自动运行 CLAM";

  const genomeStatus = genomeFile
    ? "基因组数据已解析"
    : uploadingGenome
      ? `上传中 ${genomeProgress}%`
      : "上传单患者 CSV";

  const startRightColumnResize = (startX: number) => {
    const startRight = rightWidthRef.current;
    let latestRight = startRight;

    const onMove = (event: PointerEvent) => {
      latestRight = Math.min(820, Math.max(180, startRight + startX - event.clientX));
      if (resizeFrameRef.current !== null) return;
      resizeFrameRef.current = window.requestAnimationFrame(() => {
        appShellRef.current?.style.setProperty("--right-width", `${latestRight}px`);
        resizeFrameRef.current = null;
      });
    };

    const onUp = () => {
      if (resizeFrameRef.current !== null) {
        window.cancelAnimationFrame(resizeFrameRef.current);
        resizeFrameRef.current = null;
      }
      appShellRef.current?.style.setProperty("--right-width", `${latestRight}px`);
      rightWidthRef.current = latestRight;
      setRightWidth(latestRight);
      document.body.classList.remove("is-resizing-columns");
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };

    document.body.classList.add("is-resizing-columns");
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  };

  const toggleLeftPanel = () => {
    setLeftCollapsed((value) => !value);
    setSearchOpen(false);
    setActiveCaseMenuId(null);
  };

  const toggleRightPanel = () => {
    setRightCollapsed((value) => !value);
  };

  const jumpToLatestAssistantReply = () => {
    const conversation = conversationRef.current;
    if (!conversation || !latestAssistantMessage) return;

    const target = conversation.querySelector<HTMLElement>(`[data-message-id="${latestAssistantMessage.id}"]`);
    if (target) {
      target.scrollIntoView({ behavior: "smooth", block: "start", inline: "nearest" });
      return;
    }

    conversation.scrollTo({ top: conversation.scrollHeight, behavior: "smooth" });
  };

  const effectiveLeftWidth = leftCollapsed ? LEFT_SIDEBAR_COLLAPSED_WIDTH : LEFT_SIDEBAR_WIDTH;
  const effectiveRightWidth = rightCollapsed ? 0 : rightWidth;

  const shellStyle = {
    "--left-expanded-width": `${LEFT_SIDEBAR_WIDTH}px`,
    "--left-width": `${effectiveLeftWidth}px`,
    "--right-width": `${effectiveRightWidth}px`
  } as CSSProperties;

  const visualizationKind =
    activeMessageLooksLikeTreatment || hasActiveEvidence ? "treatment" : cmtaVisualization ? "survival" : "stage";
  const visualizationHeatmapPanel = activeVisualizations.length ? (
    <section className="result-card split-result-card heatmap-card">
      <div className="result-title">
        <span>
          {cmtaVisualization ? <Dna size={16} /> : <Microscope size={16} />}
          {cmtaVisualization ? "\u8de8\u6a21\u6001\u8f6c\u6362" : "\u75c5\u7406\u70ed\u529b\u56fe"}
        </span>
      </div>
      <div className="split-card-body heatmap-panel-list">
        {cmtaVisualization ? (
          <div className="heatmap-section cross-modal-section">
            <div className="heatmap-section-title">
              <strong>{"\u5404\u7ec4\u57fa\u56e0\u5173\u6ce8\u7684\u75c5\u7406\u533a\u57df"}</strong>
            </div>
            <div className="heatmap-grid">
              {(cmtaVisualization?.heatmaps || []).map((heatmap) => (
                <article key={heatmap.sourcePath} className="heatmap-item">
                  <div className="heatmap-frame">
                    <HeatmapPreview heatmap={heatmap} onOpen={setPreviewHeatmap} />
                  </div>
                  <strong>{heatmap.label}</strong>
                  {heatmap.description ? (
                    <span className={`heatmap-mode ${heatmap.mode?.includes("negative") ? "negative" : "positive"}`}>
                      {heatmap.description}
                    </span>
                  ) : null}
                  <small title={heatmap.sourcePath}>{getFileNameFromPath(heatmap.sourcePath)}</small>
                </article>
              ))}
            </div>
            <div className="heatmap-section-title genomics-focus-title">
              <strong>{"\u75c5\u7406\u5207\u7247\u5173\u6ce8\u7684\u57fa\u56e0\u7ec4"}</strong>
            </div>
            <div className="genomics-focus-card">
              <strong>{cmtaVisualization?.genomicsFocus?.omicLabel || cmtaVisualization?.summary[0] || "\u6682\u65e0 omic \u7edf\u8ba1"}</strong>
              {cmtaVisualization?.genomicsFocus?.genes.length ? (
                <div className="gene-chip-list">
                  {(cmtaVisualization?.genomicsFocus?.genes || []).map((gene) => (
                    <span key={gene}>{gene}</span>
                  ))}
                </div>
              ) : (
                <small>{"\u5f53\u524d\u7ed3\u679c\u672a\u5305\u542b\u53ef\u5c55\u793a\u7684\u57fa\u56e0\u540d\u3002"}</small>
              )}
            </div>
          </div>
        ) : null}
        {otherVisualizations.map((visualization) => (
          <div key={visualization.id} className="heatmap-section">
            <div className="heatmap-section-title">
              <strong>{visualization.title}</strong>
            </div>
            {visualization.summary.length ? (
              <div className="heatmap-summary">
                {visualization.summary.map((line) => (
                  <span key={line}>{line}</span>
                ))}
              </div>
            ) : null}
            <div className="heatmap-grid">
              {visualization.heatmaps.map((heatmap) => (
                <article key={heatmap.sourcePath} className="heatmap-item">
                  <div className="heatmap-frame">
                    <HeatmapPreview heatmap={heatmap} onOpen={setPreviewHeatmap} />
                  </div>
                  <strong>{heatmap.label}</strong>
                  {heatmap.description ? (
                    <span className={`heatmap-mode ${heatmap.mode?.includes("negative") ? "negative" : "positive"}`}>
                      {heatmap.description}
                    </span>
                  ) : null}
                  <small title={heatmap.sourcePath}>{getFileNameFromPath(heatmap.sourcePath)}</small>
                </article>
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  ) : null;
  const effectiveExpandedSplitCard =
    visualizationHeatmapPanel && expandedSplitCard === "evidence" ? null : expandedSplitCard;
  const renderSplitExpandButton = (target: Exclude<ExpandedSplitCard, null>) => {
    const expanded = effectiveExpandedSplitCard === target;
    const label = expanded ? "\u7f29\u5c0f" : "\u653e\u5927";
    return (
      <button
        type="button"
        className="split-card-expand-btn"
        title={label}
        aria-label={label}
        onClick={() => setExpandedSplitCard(expanded ? null : target)}
      >
        {expanded ? <ZoomOut size={14} /> : <ZoomIn size={14} />}
        <span>{label}</span>
      </button>
    );
  };
  const visualizationEvidencePanel = (
    <section
      className={`result-card split-result-card evidence-card combined-evidence-card ${
        effectiveExpandedSplitCard === "evidence" ? "is-expanded-card" : ""
      }`}
    >
      <div className="result-title">
        <span>
          <BookOpenCheck size={16} />
          {"\u5f15\u7528\u8bc1\u636e"}
        </span>
        <div className="result-title-actions">
          <small>
            {activeReferencedCscoCitations.length + activeLiteratureEvidence.length
              ? `${activeReferencedCscoCitations.length + activeLiteratureEvidence.length} \u5904`
              : literatureLoading
                ? "\u52a0\u8f7d\u4e2d"
                : "\u5f85\u751f\u6210"}
          </small>
          {renderSplitExpandButton("evidence")}
        </div>
      </div>
      <div className="split-card-body combined-evidence-body">
        {activeReferencedCscoCitations.length ? (
          <div className="combined-evidence-section">
            <div className="heatmap-section-title">
              <strong>{"\u5f15\u7528CSCO\u8bc1\u636e"}</strong>
              <small>{`${activeReferencedCscoCitations.length} \u5904\u5f15\u7528`}</small>
            </div>
            <div className="csco-evidence-list">
              {activeReferencedCscoCitations.map((citation) => (
                <CscoEvidencePreview key={citation.key} citation={citation} onOpen={setPreviewCscoCitation} />
              ))}
            </div>
          </div>
        ) : null}
        <div className="combined-evidence-section">
          <div className="heatmap-section-title">
            <strong>{"\u5f15\u7528\u6587\u732e\u8bc1\u636e"}</strong>
            <small>{literatureLoading ? "\u52a0\u8f7d\u4e2d" : `${activeLiteratureEvidence.length} \u7bc7\u6587\u732e`}</small>
          </div>
          {literatureLoading && !activeLiteratureEvidence.length ? (
            <div className="knowledge-empty">
              <Loader2 className="spin" size={16} />
              {"\u6b63\u5728\u52a0\u8f7d\u5f15\u7528\u6587\u732e"}
            </div>
          ) : activeLiteratureEvidence.length ? (
            <div className="literature-list referenced-literature-list">
              {activeLiteratureEvidence.map((item) => (
                <article key={`${item.id}-${item.pmid || item.title}`} className="literature-item referenced-literature-item">
                  <div className="literature-head">
                    <strong>PubMed-{item.id}</strong>
                    <span>{item.pmid ? `PMID: ${item.pmid}` : "\u672a\u63d0\u4f9b PMID"}</span>
                  </div>
                  <h3>{item.title}</h3>
                  {item.abstract ? <p className="literature-abstract">{item.abstract}</p> : null}
                  {item.url ? (
                    <div className="literature-actions">
                      <a href={item.url} target="_blank" rel="noreferrer">
                        {"\u6253\u5f00 PubMed"}
                      </a>
                    </div>
                  ) : null}
                </article>
              ))}
            </div>
          ) : activeReferencedCscoCitations.length ? null : (
            <div className="knowledge-empty">{"\u6682\u65e0\u5f15\u7528\u8bc1\u636e"}</div>
          )}
        </div>
      </div>
    </section>
  );
  const contextPanelCollapseButton = activeDetailMessage ? (
    <button
      type="button"
      className="context-panel-collapse"
      title="返回引用证据和分析流程"
      onClick={returnToEvidenceAndFlowPanel}
    >
      <PanelRightClose size={14} />
      <span>收起</span>
    </button>
  ) : null;
  const splitPrimaryPanel = visualizationHeatmapPanel || visualizationEvidencePanel;
  const showSplitPrimaryPanel = effectiveExpandedSplitCard !== "analysis";
  const showSplitAnalysisPanel = effectiveExpandedSplitCard !== "evidence";
  const visualizationSplitPanel =
    rightPanelMode === "visualizations" ? (
      <div className={`visualization-split-layout ${visualizationKind} ${effectiveExpandedSplitCard ? "is-expanded" : ""}`}>
        {showSplitPrimaryPanel ? splitPrimaryPanel : null}
        {showSplitAnalysisPanel ? (
          <section
            className={`result-card split-result-card analysis-flow-card ${
              effectiveExpandedSplitCard === "analysis" ? "is-expanded-card" : ""
            }`}
          >
            <div className="result-title">
              <span>
                <ClipboardList size={16} />
                {"\u5206\u6790\u6d41\u7a0b"}
              </span>
              <div className="result-title-actions">{renderSplitExpandButton("analysis")}</div>
            </div>
            <AnalysisFlowChart
              log={activeAnalysisLog}
              userTask={activeUserTask}
              finalAnswer={activeDetailMessage?.content || ""}
              onOpenLiterature={openAnalysisLiteraturePanel}
              onOpenCscoRag={openAnalysisCscoRagPanel}
            />
          </section>
        ) : null}
      </div>
    ) : null;

  return (
    <div className="app-shell" style={shellStyle} ref={appShellRef}>
      <aside className={`sidebar ${leftCollapsed ? "collapsed" : ""}`} ref={sidebarRef}>
        <div className="sidebar-brand">
          {!leftCollapsed ? (
            <div className="brand-identity">
              <div className="brand-mark">
                <Stethoscope size={18} />
              </div>
              <div className="brand-copy">
                <h1>
                  <span>癌症治疗方案</span>
                  <span>生成智能体</span>
                </h1>
              </div>
            </div>
          ) : null}
          <button
            type="button"
            className="sidebar-toggle-btn"
            aria-label={leftCollapsed ? "展开左栏" : "收起左栏"}
            title={leftCollapsed ? "展开左栏" : "收起左栏"}
            onClick={toggleLeftPanel}
          >
            {leftCollapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}
          </button>
        </div>

        <button type="button" className="new-case-btn" title="新建病例" onClick={startNewCase}>
          <MessageSquarePlus size={16} />
          <span className="sidebar-btn-label">新建病例</span>
        </button>

        <button type="button" className={`knowledge-btn ${knowledgeOpen ? "active" : ""}`} title="治疗指南" onClick={openKnowledgeBase}>
          <BookOpenCheck size={16} />
          <span className="sidebar-btn-label">治疗指南</span>
        </button>

        <div className="search-popover-wrap">
          <button
            type="button"
            className="search-toggle-btn"
            title="搜索病例"
            onClick={() => {
              if (leftCollapsed) {
                setLeftCollapsed(false);
                setSearchOpen(true);
                return;
              }
              setSearchOpen((value) => !value);
            }}
          >
            <Search size={16} />
            <span className="sidebar-btn-label">搜索病例</span>
          </button>
          {searchOpen ? (
            <div className="sidebar-search-popover">
              <Search size={15} />
              <input
                ref={searchInputRef}
                value={searchQuery}
                onChange={(event) => setSearchQuery(event.target.value)}
                placeholder="搜索病例分析记录"
              />
              {searchQuery ? (
                <button type="button" aria-label="清空搜索" onClick={() => setSearchQuery("")}>
                  <XCircle size={15} />
                </button>
              ) : null}
            </div>
          ) : null}
        </div>

        {!leftCollapsed ? (
          <div className="recents">
            <div className="nav-label">
              <span>Recents</span>
              <small>{filteredCases.length}</small>
            </div>
            <div className="recent-list">
              {filteredCases.length ? (
                filteredCases.map((record) => (
                  <RecentCaseItem
                    key={record.id}
                    record={record}
                    active={record.id === currentCaseId}
                    menuOpen={activeCaseMenuId === record.id}
                    onSelect={loadCase}
                    onToggleMenu={(recordId) => setActiveCaseMenuId((value) => (value === recordId ? null : recordId))}
                    onRename={renameCase}
                    onDelete={deleteCase}
                  />
                ))
              ) : (
                <div className="empty-recents">没有匹配的病例记录</div>
              )}
            </div>
          </div>
        ) : null}
      </aside>

      <main className="interaction">
        <button
          type="button"
          className={`right-panel-edge-toggle ${rightCollapsed ? "is-collapsed" : ""}`}
          aria-label={rightCollapsed ? "展开右栏" : "收起右栏"}
          title={rightCollapsed ? "展开右栏" : "收起右栏"}
          onClick={toggleRightPanel}
        >
          {rightCollapsed ? <PanelRightOpen size={18} /> : <PanelRightClose size={18} />}
        </button>

        <div className="case-topline">
          <div>
            <strong>{getCaseTitle(patientSummary, wsiFile, genomeFile)}</strong>
          </div>
        </div>

        {errorMessage ? (
          <div className="error-banner">
            <XCircle size={16} />
            <span>{errorMessage}</span>
          </div>
        ) : null}

        <div className="conversation" ref={conversationRef}>
          {messages.length ? (
            <div className="message-stack">
              {messages.map((message) => {
                const visualizations = visualizationsByMessageId.get(message.id) || [];
                return (
                  <ChatBubble
                    key={message.id}
                    message={message}
                    visualizations={visualizations}
                    active={activeVisualizationMessageId === message.id}
                    onCopy={(content) => {
                      void copyTextToClipboard(content);
                    }}
                    onRetry={() => retryAssistantMessage(message.id)}
                    retryDisabled={agentBusy}
                    onOpenVisualizations={() => {
                      const nextOpen = activeVisualizationMessageId !== message.id;
                      setActiveVisualizationMessageId(nextOpen ? message.id : null);
                      if (nextOpen) setRightCollapsed(false);
                      setKnowledgeOpen(false);
                      setLiteratureOpen(false);
                      setCscoRagOpen(false);
                      setContextLiteratureText("");
                    }}
                  />
                );
              })}
            </div>
          ) : (
            <div className="welcome-panel">
              <Sparkles size={22} />
              <h2>我能为你做些什么？</h2>
              <p>请输入您的问题并点击发送获取智能体回复</p>
              <div className="quick-grid">
                {quickTasks.map((task) => {
                  const Icon = task.icon;
                  return (
                    <button key={task.title} type="button" onClick={() => setQuery(task.prompt)}>
                      <Icon size={18} />
                      <strong>{task.title}</strong>
                      <span>{task.desc}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {latestAssistantMessage ? (
          <button
            type="button"
            className="jump-latest-reply-btn"
            aria-label="跳转到智能体最新回复"
            title="跳转到智能体最新回复"
            onClick={jumpToLatestAssistantReply}
          >
            <ArrowDown size={24} strokeWidth={2.7} />
          </button>
        ) : null}

        <div className="composer-shell">
          <div className="prompt-box">
            <input
              ref={pathologyInputRef}
              type="file"
              accept=".svs,.tif,.tiff"
              disabled={agentBusy || clamBusy || uploadingWsi}
              onChange={(event) => {
                const selected = event.target.files?.[0];
                if (selected) void handleWsiUpload(selected);
                event.currentTarget.value = "";
              }}
            />
            <input
              ref={genomeInputRef}
              type="file"
              accept=".csv"
              disabled={agentBusy || uploadingGenome}
              onChange={(event) => {
                const selected = event.target.files?.[0];
                if (selected) void handleGenomeUpload(selected);
                event.currentTarget.value = "";
              }}
            />

            <textarea
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void submitQuery();
                }
              }}
              placeholder="Reply..."
              disabled={agentBusy}
            />

            <div className="composer-actions">
              <div className="attach-area">
                <button
                  type="button"
                  className="plus-btn"
                  aria-label="上传数据"
                  disabled={agentBusy || clamBusy}
                  onClick={() => {
                    setUploadMenuOpen((value) => !value);
                    setModelMenuOpen(false);
                  }}
                >
                  <Plus size={18} />
                </button>
                {uploadMenuOpen ? (
                  <div className="upload-popover">
                    <button
                      type="button"
                      disabled={agentBusy || clamBusy || uploadingWsi}
                      onClick={() => pathologyInputRef.current?.click()}
                    >
                      <Microscope size={15} />
                      <span>
                        病理数据上传
                        <small>{pathologyStatus}</small>
                      </span>
                    </button>
                    <button
                      type="button"
                      disabled={agentBusy || uploadingGenome}
                      onClick={() => genomeInputRef.current?.click()}
                    >
                      <Dna size={15} />
                      <span>
                        基因组数据上传
                        <small>{genomeStatus}</small>
                      </span>
                    </button>
                  </div>
                ) : null}

                {(wsiFile || uploadingWsi || clamBusy) ? (
                  <div className={`attachment-chip ${hasClamFeature ? "done" : ""}`}>
                    {uploadingWsi || clamBusy ? <Loader2 className="spin" size={13} /> : <FileArchive size={13} />}
                    <span>{wsiFile ? wsiFile.name : "病理数据"}</span>
                    {hasClamFeature ? <CheckCircle2 size={13} /> : <small>{clamBusy ? `${clamProgress}%` : `${pathologyProgress}%`}</small>}
                  </div>
                ) : null}

                {(genomeFile || uploadingGenome) ? (
                  <div className={`attachment-chip ${genomeFile ? "done" : ""}`}>
                    {uploadingGenome ? <Loader2 className="spin" size={13} /> : <Dna size={13} />}
                    <span>{genomeFile ? genomeFile.name : "基因组数据"}</span>
                    {genomeFile ? <CheckCircle2 size={13} /> : <small>{genomeProgress}%</small>}
                  </div>
                ) : null}
              </div>

              <div className="composer-right">
                <div className="model-menu-wrap">
                  <button
                    type="button"
                    className={`model-menu-trigger ${modelMenuOpen ? "active" : ""}`}
                    disabled={agentBusy}
                    onClick={() => {
                      setModelMenuOpen((value) => !value);
                      setUploadMenuOpen(false);
                    }}
                  >
                    <span>{modelLabel(llmModel)}</span>
                    <ChevronRight size={15} />
                  </button>
                  {modelMenuOpen ? (
                    <div className="model-menu">
                      {llmModelOptions.map((option) => (
                        <button
                          key={option.value}
                          type="button"
                          className={option.value === llmModel ? "active" : ""}
                          onClick={() => {
                            setLlmModel(option.value);
                            setModelMenuOpen(false);
                          }}
                        >
                          <span>{option.label}</span>
                          {option.value === llmModel ? <CheckCircle2 size={14} /> : null}
                        </button>
                      ))}
                    </div>
                  ) : null}
                </div>
                <button type="button" className="send-btn" disabled={!query.trim() || agentBusy} onClick={submitQuery}>
                  {agentBusy ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
                </button>
              </div>
            </div>
          </div>
        </div>
      </main>

      <div
        className={`column-resizer right-resizer ${rightCollapsed ? "is-disabled" : ""}`}
        role="separator"
        aria-label="调整右栏宽度"
        onPointerDown={(event) => {
          if (!rightCollapsed) startRightColumnResize(event.clientX);
        }}
      />

      <aside
        className={`result-panel ${showRightContent ? "" : "is-empty"} ${rightCollapsed ? "collapsed" : ""} ${
          rightPanelMode === "visualizations" ? "is-visualization-mode" : ""
        }`}
      >
        {!rightCollapsed && showRightContent ? (
          <>
            {rightPanelMode === "knowledge" ? (
              <section className="result-card">
                <div className="result-title">
                  <span>
                    <BookOpenCheck size={16} />
                    CSCO 治疗指南
                  </span>
                  <div className="result-title-actions">
                    <button
                      type="button"
                      className="knowledge-upload-btn"
                      title={knowledgeUploading ? "上传中" : "上传 PDF"}
                      aria-label={knowledgeUploading ? "上传中" : "上传 PDF"}
                      disabled={knowledgeLoading || knowledgeUploading}
                      onClick={() => knowledgeInputRef.current?.click()}
                    >
                      {knowledgeUploading ? <Loader2 className="spin" size={14} /> : <Plus size={14} />}
                    </button>
                  </div>
                </div>
                <input
                  ref={knowledgeInputRef}
                  className="knowledge-file-input"
                  type="file"
                  accept=".pdf,application/pdf"
                  disabled={knowledgeLoading || knowledgeUploading}
                  onChange={(event) => {
                    const selected = event.target.files?.[0];
                    if (selected) void handleKnowledgeUpload(selected);
                    event.currentTarget.value = "";
                  }}
                />
                <div className="knowledge-list">
                  {knowledgeLoading ? (
                    <div className="knowledge-empty">
                      <Loader2 className="spin" size={16} />
                      正在加载治疗指南
                    </div>
                  ) : knowledgeFiles.length ? (
                    knowledgeFiles.map((file) => {
                      const deleting = knowledgeDeletingName === file.name;
                      return (
                        <div className="knowledge-file-row" key={file.name}>
                          <button
                            type="button"
                            className="knowledge-open-btn"
                            disabled={Boolean(knowledgeDeletingName)}
                            onClick={() => window.open(absoluteApiUrl(file.url), "_blank", "noopener,noreferrer")}
                          >
                            <FileText size={15} />
                            <span>
                              <strong>{file.name}</strong>
                              <small>{formatFileSize(file.size)}</small>
                            </span>
                          </button>
                          <button
                            type="button"
                            className="knowledge-delete-btn"
                            title="删除文件"
                            aria-label={`删除${file.name}`}
                            disabled={Boolean(knowledgeDeletingName) || knowledgeUploading}
                            onClick={() => void deleteKnowledgeFile(file)}
                          >
                            {deleting ? <Loader2 className="spin" size={14} /> : <Trash2 size={14} />}
                          </button>
                        </div>
                      );
                    })
                  ) : (
                    <div className="knowledge-empty">未找到 CSCO PDF 文件</div>
                  )}
                </div>
              </section>
            ) : null}

            {rightPanelMode === "literature" ? (
              <section className="result-card">
                <div className="result-title">
                  <span>
                    <FileText size={16} />
                    本病例检索文献
                  </span>
                  <div className="result-title-actions">
                    <small>
                      {literatureLoading
                        ? "加载中"
                        : panelLiteratureItems.length
                          ? `${panelLiteratureItems.length} 篇文献`
                          : panelLiteratureText
                            ? "已获取"
                            : "待生成"}
                    </small>
                    {contextPanelCollapseButton}
                  </div>
                </div>
                {literatureLoading ? (
                  <div className="knowledge-empty">
                    <Loader2 className="spin" size={16} />
                    正在加载检索文献
                  </div>
                ) : panelLiteratureItems.length ? (
                  <div className="literature-list">
                    {panelLiteratureItems.map((item) => (
                      <article key={`${item.id}-${item.pmid || item.title}`} className="literature-item">
                        <div className="literature-head">
                          <strong>PubMed-{item.id}</strong>
                          <span>{item.pmid ? `PMID: ${item.pmid}` : "PMID 未提供"}</span>
                        </div>
                        <h3>{item.title}</h3>
                        <div className="literature-meta">
                          {item.year ? <span>年份: {item.year}</span> : null}
                          {item.authors ? <span>作者: {item.authors}</span> : null}
                          {item.studyType ? <span>类型: {item.studyType}</span> : null}
                        </div>
                        {item.journal ? <p className="literature-journal">{item.journal}</p> : null}
                        {item.abstract ? <p className="literature-abstract">{item.abstract}</p> : null}
                        <div className="literature-actions">
                          {item.citation ? <span>{item.citation}</span> : null}
                          {item.url ? (
                            <a href={item.url} target="_blank" rel="noreferrer">
                              打开 PubMed
                            </a>
                          ) : null}
                        </div>
                      </article>
                    ))}
                  </div>
                ) : panelLiteratureText ? (
                  <pre className="literature-output">{panelLiteratureText}</pre>
                ) : (
                  <div className="knowledge-empty">治疗方案生成后显示 PubMed 检索文献</div>
                )}
              </section>
            ) : null}

            {rightPanelMode === "csco-rag" ? (
              <section className="result-card evidence-card">
                <div className="result-title">
                  <span>
                    <BookOpenCheck size={16} />
                    本病例 CSCO RAG
                  </span>
                  <div className="result-title-actions">
                    <small>{cscoRagEvidenceItems.length ? `${cscoRagEvidenceItems.length} 个片段` : "待生成"}</small>
                    {contextPanelCollapseButton}
                  </div>
                </div>
                {cscoRagEvidenceItems.length ? (
                  <div className="csco-rag-snippet-list">
                    {cscoRagEvidenceItems.map((item) => (
                      <div key={item.key} className="csco-rag-snippet-item">
                        <CscoEvidencePreview citation={item} onOpen={setPreviewCscoCitation} />
                        {item.content ? <pre className="csco-rag-snippet-text">{item.content}</pre> : null}
                      </div>
                    ))}
                  </div>
                ) : activeCscoRagEvidenceText ? (
                  <pre className="literature-output">{activeCscoRagEvidenceText}</pre>
                ) : (
                  <div className="knowledge-empty">治疗方案生成后显示本病例 RAG 检索片段</div>
                )}
              </section>
            ) : null}

            {visualizationSplitPanel}

            {false && rightPanelMode === "visualizations" && activeReferencedCscoCitations.length ? (
              <section className="result-card evidence-card">
                <div className="result-title">
                  <span>
                    <BookOpenCheck size={16} />
                    引用CSCO证据
                  </span>
                  <small>{activeReferencedCscoCitations.length} 处引用</small>
                </div>
                <div className="csco-evidence-list">
                  {activeReferencedCscoCitations.map((citation) => (
                    <CscoEvidencePreview key={citation.key} citation={citation} onOpen={setPreviewCscoCitation} />
                  ))}
                </div>
              </section>
            ) : null}

            {false && rightPanelMode === "visualizations" &&
            (activeLiteratureEvidence.length || (activeMessageLooksLikeTreatment && literatureLoading)) ? (
              <section className="result-card evidence-card">
                <div className="result-title">
                  <span>
                    <FileText size={16} />
                    引用文献证据
                  </span>
                  <small>{literatureLoading ? "加载中" : `${activeLiteratureEvidence.length} 篇文献`}</small>
                </div>
                {literatureLoading && !activeLiteratureEvidence.length ? (
                  <div className="knowledge-empty">
                    <Loader2 className="spin" size={16} />
                    正在加载引用文献
                  </div>
                ) : (
                  <div className="literature-list referenced-literature-list">
                    {activeLiteratureEvidence.map((item) => (
                      <article key={`${item.id}-${item.pmid || item.title}`} className="literature-item referenced-literature-item">
                        <div className="literature-head">
                          <strong>PubMed-{item.id}</strong>
                          <span>{item.pmid ? `PMID: ${item.pmid}` : "PMID 未提供"}</span>
                        </div>
                        <h3>{item.title}</h3>
                        {item.abstract ? <p className="literature-abstract">{item.abstract}</p> : null}
                        {item.url ? (
                          <div className="literature-actions">
                            <a href={item.url} target="_blank" rel="noreferrer">
                              打开 PubMed
                            </a>
                          </div>
                        ) : null}
                      </article>
                    ))}
                  </div>
                )}
              </section>
            ) : null}

            {false && rightPanelMode === "visualizations" && activeVisualizations.length ? (
              <section className="result-card heatmap-card">
                <div className="result-title">
                  <span>
                    {cmtaVisualization ? <Dna size={16} /> : <Microscope size={16} />}
                    {cmtaVisualization ? "跨模态转换" : "病理热力图"}
                  </span>
                </div>
                <div className="heatmap-panel-list">
                  {cmtaVisualization ? (
                    <div className="heatmap-section cross-modal-section">
                      <div className="heatmap-section-title">
                        <strong>各组基因关注的病理区域</strong>
                      </div>
                      <div className="heatmap-grid">
                        {(cmtaVisualization?.heatmaps || []).map((heatmap) => (
                          <article key={heatmap.sourcePath} className="heatmap-item">
                            <div className="heatmap-frame">
                              <HeatmapPreview heatmap={heatmap} onOpen={setPreviewHeatmap} />
                            </div>
                            <strong>{heatmap.label}</strong>
                            {heatmap.description ? (
                              <span className={`heatmap-mode ${heatmap.mode?.includes("negative") ? "negative" : "positive"}`}>
                                {heatmap.description}
                              </span>
                            ) : null}
                            <small title={heatmap.sourcePath}>{getFileNameFromPath(heatmap.sourcePath)}</small>
                          </article>
                        ))}
                      </div>
                      <div className="heatmap-section-title genomics-focus-title">
                        <strong>病理切片关注的基因组</strong>
                      </div>
                      <div className="genomics-focus-card">
                        <strong>{cmtaVisualization?.genomicsFocus?.omicLabel || cmtaVisualization?.summary[0] || "暂无 omic 统计"}</strong>
                        {cmtaVisualization?.genomicsFocus?.genes.length ? (
                          <div className="gene-chip-list">
                            {(cmtaVisualization?.genomicsFocus?.genes || []).map((gene) => (
                              <span key={gene}>{gene}</span>
                            ))}
                          </div>
                        ) : (
                          <small>当前结果未包含可展示的基因名。</small>
                        )}
                      </div>
                    </div>
                  ) : null}
                  {otherVisualizations.map((visualization) => (
                    <div key={visualization.id} className="heatmap-section">
                      <div className="heatmap-section-title">
                        <strong>{visualization.title}</strong>
                      </div>
                      {visualization.summary.length ? (
                        <div className="heatmap-summary">
                          {visualization.summary.map((line) => (
                            <span key={line}>{line}</span>
                          ))}
                        </div>
                      ) : null}
                      <div className="heatmap-grid">
                        {visualization.heatmaps.map((heatmap) => (
                          <article key={heatmap.sourcePath} className="heatmap-item">
                            <div className="heatmap-frame">
                              <HeatmapPreview heatmap={heatmap} onOpen={setPreviewHeatmap} />
                            </div>
                            <strong>{heatmap.label}</strong>
                            {heatmap.description ? (
                              <span className={`heatmap-mode ${heatmap.mode?.includes("negative") ? "negative" : "positive"}`}>
                                {heatmap.description}
                              </span>
                            ) : null}
                            <small title={heatmap.sourcePath}>{getFileNameFromPath(heatmap.sourcePath)}</small>
                          </article>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            ) : null}

            {false && rightPanelMode === "visualizations" ? (
              <section className="result-card analysis-flow-card">
                <div className="result-title">
                  <span>
                    <ClipboardList size={16} />
                    分析流程
                  </span>
                </div>
                <AnalysisFlowChart
                  log={activeAnalysisLog}
                  userTask={activeUserTask}
                  finalAnswer={activeDetailMessage?.content || ""}
                  onOpenLiterature={openAnalysisLiteraturePanel}
                  onOpenCscoRag={openAnalysisCscoRagPanel}
                />
              </section>
            ) : null}

            {showDefaultRightContent && Object.keys(patientSummary).length ? (
              <section className="result-card">
                <div className="result-title">
                  <span>
                    <FileText size={16} />
                    患者信息
                  </span>
                </div>
                <div className="patient-table">
                  {Object.entries(patientSummary).map(([label, value]) => (
                    <div key={label}>
                      <span>{label}</span>
                      <strong>{value}</strong>
                    </div>
                  ))}
                </div>
              </section>
            ) : null}

            {showDefaultRightContent && genomeColumns.length ? (
              <section className="result-card">
                <div className="result-title">
                  <span>
                    <Dna size={16} />
                    基因组数据
                  </span>
                </div>
                <div className="genome-column-list">
                  {genomeColumns.map((column) => (
                    <span key={column}>{column}</span>
                  ))}
                </div>
              </section>
            ) : null}

            {showDefaultRightContent && (wsiFile || pathologyJpgUrl || stage === "clam-running") ? (
              <section className="result-card">
                <div className="result-title">
                  <span>
                    <Microscope size={16} />
                    病理图像
                  </span>
                  {hasClamFeature ? <CheckCircle2 size={16} /> : null}
                </div>
                <div className="pathology-preview">
                  {pathologyJpgUrl ? (
                    <PathologyMaskPreview imageUrl={pathologyJpgUrl} onOpen={() => setPathologyPreviewOpen(true)} />
                  ) : (
                    <div>
                      {stage === "clam-running" ? <Loader2 className="spin" size={20} /> : <Microscope size={20} />}
                      <span>{stage === "clam-running" ? `CLAM 处理中 ${clamProgress}%` : "等待 CLAM 输出 tissue mask"}</span>
                    </div>
                  )}
                </div>
                {wsiFile ? <p className="file-line">{wsiFile.name}</p> : null}
                {pathologyPt ? <p className="file-line ok-text">PT: {pathologyPt}</p> : null}
              </section>
            ) : null}

            {showDefaultRightContent && (sessionId || genomeFile || pathologyPt) ? (
              <section className="result-card compact">
                <div className="result-title">
                  <span>
                    <PanelRight size={16} />
                    病例状态
                  </span>
                </div>
                <div className="status-list">
                  <div>
                    <Clock3 size={14} />
                    <span>{stageLabels[stage]}</span>
                  </div>
                  <div>
                    <Sparkles size={14} />
                    <span>大模型: {modelLabel(llmModel)}</span>
                  </div>
                  {sessionId ? (
                    <div>
                      <PanelRight size={14} />
                      <span>Session: {sessionId}</span>
                    </div>
                  ) : null}
                  {genomeFile ? (
                    <div>
                      <Dna size={14} />
                      <span>{genomeFile.name}</span>
                    </div>
                  ) : null}
                </div>
              </section>
            ) : null}
          </>
        ) : null}
      </aside>
      {previewHeatmap ? <HeatmapLightbox heatmap={previewHeatmap} onClose={() => setPreviewHeatmap(null)} /> : null}
      {previewCscoCitation?.imageUrl ? (
        <ZoomableImageLightbox
          title={`CSCO-${previewCscoCitation.id}`}
          subtitle={`${previewCscoCitation.filename} · ${previewCscoCitation.pageLabel}`}
          imageUrl={previewCscoCitation.imageUrl}
          alt={`${previewCscoCitation.filename} ${previewCscoCitation.pageLabel}`}
          dialogLabel="CSCO证据预览图"
          onClose={() => setPreviewCscoCitation(null)}
        />
      ) : null}
      {pathologyPreviewOpen && pathologyJpgUrl ? (
        <ZoomableImageLightbox
          title="CLAM mask 图像"
          subtitle={wsiFile?.name || "病理图像"}
          imageUrl={pathologyJpgUrl}
          alt="CLAM tissue mask"
          dialogLabel="CLAM mask 图像预览"
          onClose={() => setPathologyPreviewOpen(false)}
        />
      ) : null}
    </div>
  );
}
