"""
医学任务ReAct智能体。
负责围绕当前病例编排分期预测、生存预测和治疗方案生成。
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import HumanMessage, SystemMessage

import config
import medical_tools
from medical_tools import (
    TOOL_MAP,
    generate_treatment_plan,
    get_active_llm_model,
    predict_survival,
    predict_tnm_stage,
    search_literature,
    set_active_llm_model,
    _invoke_llm_with_retry,
    _strip_think_tags,
    _to_text,
)

# ==================== 医学任务ReAct Agent Prompt ====================

REACT_SYSTEM_PROMPT = """你是一个医学任务ReAct Agent，只负责当前病例的三类医学任务：分期预测、生存时间/风险预测、治疗方案生成。

你不是日常问答助手。普通问答由主智能体直接处理；你只在主智能体已经判断用户需要当前病例医学任务时被调用。

## 你拥有以下工具（必须严格按照以下格式调用）：

### 工具列表
1. **predict_tnm_stage**: 预测TNM分期
   - 参数: features_input (str), cancer_type (str)
   - 能力: 基于病理特征文件返回T/N/M和临床分期，并可能返回病理热力图路径

2. **predict_survival**: 预测生存时间、风险等级，并返回CMTA跨模态attention统计和6组omic病理热力图路径
   - 参数: features_input (str), gene_input (str), cancer_type (str)
   - 能力: 基于病理特征文件和基因组CSV返回预测生存时间、风险等级、风险分数和跨模态可视化路径

3. **search_literature**: 搜索PubMed文献
   - 参数: cancer_type (str), stage (str，可选), treatment (str，可选), keywords (str，可选), max_results (int，默认8，范围5-10), recent_years (int，默认5), min_impact_factor (float，默认10)
   - 能力: 检索近年、期刊影响因子达到阈值并按年份从近到远排序的5-10篇癌症治疗相关PubMed证据，返回可引用PMID

4. **generate_treatment_plan**: 生成最终治疗方案
   - 参数: cancer_type (str), t (str), n (str), m (str), stage (str), risk_level (str), survival_months (str), literature_results (str，短PMID摘要即可)
   - 能力: 基于患者分期/预后信息和PubMed证据生成治疗方案；该工具内部会先RAG检索CSCO，并从agent内部状态读取完整PubMed证据生成最终方案
   - 前置条件: 必须具备T/N/M、临床分期、预测生存时间、风险等级，以及包含PMID的PubMed检索证据

## 决策规则
1、你需要通过思考(Thought)、行动(Action)、观察(Observation)的循环来解决问题。每一次循环采用思考(Thought)、行动(Action)、观察(Observation)的格式并且只调用一个工具。
2、首先分析用户目标、已有患者信息、可用文件路径和工具能力，自主决定下一步是否调用工具以及调用哪个工具。
3、只调用对当前用户目标必要的工具；不要为了展示流程而调用无关工具。
4、如果某个工具缺少必要输入，不要编造路径或参数，应说明缺少什么，并根据已有信息选择可行下一步。
5、如果选择调用generate_treatment_plan，必须保证其literature_results来自有效PubMed检索结果且包含PMID；不得凭空构造PubMed证据，也不要在Action Input里复述PubMed全文。
6、generate_treatment_plan会强制进行CSCO RAG检索；如果最终回答包含治疗方案，必须保留该工具返回的CSCO页码和PubMed PMID。
7、工具返回后，不能编造或改写关键预测值、CSCO页码或PMID。然后基于工具结果进行新一轮思考（Thought），决定下一步行动。
8、当Observation已经足以回答用户当前医学任务时，输出Final Answer；如果治疗方案已经由generate_treatment_plan返回，最终答案必须使用该工具返回内容。
9、重复上述过程，直到完成用户当前医学任务，或达到最大循环次数。

## 输出格式

**调用工具时**:
Thought: 我现在有什么信息，当前医学任务还缺少什么，下一步调用哪个工具
Action: 工具名
Action Input: {"参数名": "参数值"}
```

**完成时**:
Final Answer: 最终回答用户当前医学任务；如果是治疗方案任务，则必须原样输出generate_treatment_plan返回的最终治疗方案
```
"""

# ==================== 医学任务ReAct类 ====================

class MedicalTaskReActAgent:
    """分期、生存和治疗方案三类医学任务的ReAct Agent。"""
    
    def __init__(self, verbose: bool = True, max_iterations: int = 10, llm_model: str | None = None):
        self.verbose = verbose
        self.max_iterations = max_iterations
        if llm_model:
            set_active_llm_model(llm_model)
        self.llm = ChatZhipuAI(
            model=get_active_llm_model(),
            temperature=0.1  # 降低温度以保持稳定决策
        )
        self.chat_history = []
        self.reasoning_steps = []
        self.tool_call_records = []
        self.last_literature_results = ""
        self.last_rag_retrieved = ""
        self.last_model_predictions = ""
        self.last_treatment_plan = ""
        self.final_answer = ""
        self.patient_info = ""
        self.file_paths = {}
    
    def _extract_file_paths(self, user_input: str) -> Dict[str, str]:
        """从用户输入提取文件路径"""
        paths = {}
        
        # 提取特征文件
        match = re.search(r'病理特征文件:\s*([^\s，,]+\.(?:pt|npy))', user_input, re.I)
        if not match:
            match = re.search(r'([\w/\\:().\-]+[\w/\\().\-]*\.(?:pt|npy))', user_input, re.I)
        if match:
            paths['features_input'] = match.group(1)
        
        # 提取基因文件
        match = re.search(r'基因组数据文件:\s*([^\s，,]+\.csv)', user_input, re.I)
        if match:
            paths['gene_input'] = match.group(1)
        
        return paths

    def _format_action_input(self,tool_input: Any) -> str:
        if isinstance(tool_input, dict):
            return json.dumps(tool_input, ensure_ascii=False)
        return str(tool_input)

    def _extract_thought(self,llm_response: str) -> str:
        llm_response = _strip_think_tags(llm_response)
        match = re.search(r'Thought:\s*(.*?)(?:\nAction:|\Z)', llm_response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return re.split(r'\nAction:\s*\w+', llm_response, maxsplit=1)[0].strip()

    def _compact_text(self, text: Any, max_chars: int = 1200) -> str:
        text = _to_text(text).strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + f"\n...（已压缩，原始长度{len(text)}字符）"

    def _extract_observation_field(self, observation: str, patterns: List[str]) -> str:
        for pattern in patterns:
            match = re.search(pattern, observation, re.I)
            if match:
                return match.group(1).strip()
        return ""

    def _summarize_prediction_observation(self, action: str, observation: str) -> str:
        if action == "predict_tnm_stage":
            fields = [
                ("T分期", [r"T分期:\s*([^,，\n]+)"]),
                ("N分期", [r"N分期:\s*([^,，\n]+)"]),
                ("M分期", [r"M分期:\s*([^,，\n]+)"]),
                ("临床分期", [r"临床分期:\s*([^,，\n]+)"]),
            ]
            title = "predict_tnm_stage已返回分期预测结果。"
        else:
            fields = [
                ("预测生存时间", [r"预测生存时间:\s*([^,，\n]+(?:个月)?)"]),
                ("风险等级", [r"风险等级:\s*([^,，\n]+)"]),
                ("风险评分", [r"风险(?:评分|分数):\s*([^,，\n]+)"]),
            ]
            title = "predict_survival已返回生存时间和风险预测结果。"

        lines = [title]
        for label, patterns in fields:
            value = self._extract_observation_field(observation, patterns)
            if value:
                lines.append(f"- {label}: {value}")

        if len(lines) == 1:
            lines.append(self._compact_text(observation, 900))
        lines.append("完整预测结果已保存在agent状态中，后续调用generate_treatment_plan时会自动补全所需字段。")
        return "\n".join(lines)

    def _summarize_literature_observation(self, observation: str) -> str:
        observation = _to_text(observation)
        pmids = []
        for pmid in re.findall(r"PMID:\s*(\d+)", observation):
            if pmid not in pmids:
                pmids.append(pmid)

        items = []
        pattern = r"【PubMed-(\d+)\s*\|\s*PMID:\s*(\d+)\s*\|\s*年份:\s*([^】]+)】\s*\n题名:\s*(.*?)(?:\n|$)"
        for match in re.finditer(pattern, observation, re.S):
            idx, pmid, year, title = match.groups()
            title = re.sub(r"\s+", " ", title).strip()
            items.append(f"- PubMed-{idx}: PMID {pmid}, {year}, {self._compact_text(title, 180)}")
            if len(items) >= 5:
                break

        lines = ["search_literature已返回有效PubMed证据。"]
        if pmids:
            lines.append(f"- 可用PMID: {', '.join(pmids[:10])}")
        if items:
            lines.append("- 文献条目摘要:")
            lines.extend(items)
        else:
            lines.append(self._compact_text(observation, 900))
        lines.append("完整PubMed证据已保存在agent状态中；调用generate_treatment_plan时系统会通过内部上下文传入完整证据，Action Input只需PMID短摘要。")
        return "\n".join(lines)

    def _format_action_input_for_history(self, action: str, action_input: Any) -> str:
        if not isinstance(action_input, dict):
            return self._compact_text(action_input, 1000)

        compact_input = {}
        for key, value in action_input.items():
            if key == "literature_results":
                compact_input[key] = self._summarize_literature_observation(value)
            elif isinstance(value, str) and len(value) > 500:
                compact_input[key] = self._compact_text(value, 500)
            else:
                compact_input[key] = value
        return json.dumps(compact_input, ensure_ascii=False)

    def _format_observation_for_record(self, action: str, observation: str) -> str:
        observation = _to_text(observation)
        if action == "generate_treatment_plan":
            return (
                "generate_treatment_plan工具已完成执行。"
                "CSCO RAG证据、PubMed证据和最终治疗方案详见下方【最终治疗方案】。"
            )
        if action in {"predict_tnm_stage", "predict_survival"}:
            return self._summarize_prediction_observation(action, observation)
        if action == "search_literature":
            return self._summarize_literature_observation(observation)
        return self._compact_text(observation, 1200)

    def _has_valid_literature(self,literature_results: str) -> bool:
        text = _to_text(literature_results)
        if not text:
            return False
        invalid_markers = ["搜索出错", "文献搜索失败", "未找到相关文献", "未找到可解析"]
        return bool(medical_tools._coerce_pmids(text)) and not any(marker in text for marker in invalid_markers)

    def _literature_detail_score(self, literature_results: Any) -> int:
        text = _to_text(literature_results)
        if not text.strip():
            return 0
        score = 0
        if "PMID:" in text:
            score += 1
        if "【PubMed-" in text or "PubMed-" in text:
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
        score += sum(1 for marker in detail_markers if marker in text)
        return score

    def _remember_literature_results(self, literature_results: Any) -> None:
        candidate = _to_text(literature_results).strip()
        if not candidate:
            return
        current_score = self._literature_detail_score(self.last_literature_results)
        candidate_score = self._literature_detail_score(candidate)
        if candidate_score > current_score or (candidate_score == current_score and len(candidate) > len(self.last_literature_results)):
            self.last_literature_results = candidate

    #构建治疗方案工具输入时的文献参数
    def _build_literature_status_input(self) -> str:
        pmids = medical_tools._extract_pmids(self.last_literature_results)
        if not pmids:
            return ""
        return "已有有效PubMed证据，PMID: " + ", ".join(pmids)

    def _is_missing_tool_value(self, value: Any) -> bool:
        text = _to_text(value).strip()
        return not text or text.lower() in {"n/a", "na", "none", "null"} or text in {"无", "未提供", "未知"}

    def _is_treatment_request(self, user_input: str) -> bool:
        text = _to_text(user_input).lower()
        return any(keyword in text for keyword in ["治疗", "方案", "treatment"])

    def _task_goal_label(self, task_goal: str | None) -> str:
        labels = {
            "stage": "TNM/临床分期预测",
            "survival": "生存时间/风险预测",
            "treatment": "个体化治疗方案生成",
        }
        return labels.get(task_goal or "", "由用户当前输入决定")

    def _tool_satisfies_task_goal(self, task_goal: str | None, tool_name: str) -> bool:
        return (
            (task_goal == "stage" and tool_name == "predict_tnm_stage")
            or (task_goal == "survival" and tool_name == "predict_survival")
            or (task_goal == "treatment" and tool_name == "generate_treatment_plan")
        )

    #提取预测模型结果
    def _extract_prediction_field(self, pattern: str) -> str:
        match = re.search(pattern, self.last_model_predictions)
        return match.group(1).strip() if match else ""
   
    #设置治疗方案生成工具输入参数
    def _build_treatment_tool_input(self, cancer_type: str) -> Dict[str, Any]:
        available_pmids = medical_tools._extract_pmids(self.last_literature_results)
        return {
            "cancer_type": cancer_type,
            "t": self._extract_prediction_field(r"T分期:\s*([^,，\n]+)"),
            "n": self._extract_prediction_field(r"N分期:\s*([^,，\n]+)"),
            "m": self._extract_prediction_field(r"M分期:\s*([^,，\n]+)"),
            "stage": self._extract_prediction_field(r"临床分期:\s*([^,，\n]+)"),
            "survival_months": self._extract_prediction_field(r"预测生存时间:\s*([0-9.]+)\s*个月"),
            "risk_level": self._extract_prediction_field(r"风险等级:\s*([^,，\n]+)"),
            "literature_results": self._build_literature_status_input(),
            "available_pmids": ", ".join(available_pmids),
        }

    def _fill_treatment_tool_input(self, tool_input: Dict[str, Any], cancer_type: str) -> Dict[str, Any]:
        predicted = self._build_treatment_tool_input(cancer_type)
        for key, value in predicted.items():
            if key in {"available_pmids", "literature_results"} and value:
                tool_input[key] = value
            elif value and not _to_text(tool_input.get(key)).strip():
                tool_input[key] = value
        return tool_input

    def _missing_treatment_tool_inputs(self, tool_input: Dict[str, Any]) -> List[str]:
        required_fields = {
            "t": "T分期",
            "n": "N分期",
            "m": "M分期",
            "stage": "临床分期",
            "survival_months": "预测生存时间",
            "risk_level": "风险等级",
        }
        missing = [
            label
            for key, label in required_fields.items()
            if self._is_missing_tool_value(tool_input.get(key))
        ]
        if not self._has_valid_literature(tool_input.get("literature_results", "")):
            missing.append("有效PubMed文献证据")
        return missing

    def _build_treatment_readiness_context(self, cancer_type: str) -> str:
        tool_input = self._fill_treatment_tool_input({}, cancer_type)
        missing = self._missing_treatment_tool_inputs(tool_input)
        literature_status = "已有有效PubMed证据" if self._has_valid_literature(tool_input.get("literature_results", "")) else "缺少有效PubMed证据"
        pmids = []
        for pmid in re.findall(r"PMID:\s*(\d+)", _to_text(tool_input.get("literature_results", ""))):
            if pmid not in pmids:
                pmids.append(pmid)

        lines = [
            "当前generate_treatment_plan输入状态:",
            f"- cancer_type: {tool_input.get('cancer_type') or cancer_type}",
            f"- T/N/M: {tool_input.get('t') or '缺少'} / {tool_input.get('n') or '缺少'} / {tool_input.get('m') or '缺少'}",
            f"- 临床分期: {tool_input.get('stage') or '缺少'}",
            f"- 预测生存时间: {tool_input.get('survival_months') or '缺少'}",
            f"- 风险等级: {tool_input.get('risk_level') or '缺少'}",
            f"- PubMed证据: {literature_status}" + (f"，PMID: {', '.join(pmids[:10])}" if pmids else ""),
        ]
        if missing:
            lines.append("尚缺少: " + "、".join(missing))
        else:
            lines.append("治疗方案工具所需输入已齐；如果当前目标仍是生成治疗方案，下一步可调用generate_treatment_plan。")
            lines.append("Action Input只需包含必要短字段；完整PubMed证据会通过内部上下文传给治疗方案工具，不要把PubMed全文写进上下文。")
        return "\n".join(lines)

    def _build_followup_context(
        self,
        tool_name: str,
        tool_result: str,
        cancer_type: str,
        task_goal: str | None,
    ) -> str:
        summary = self._format_observation_for_record(tool_name, tool_result)
        lines = [
            f"上一轮工具 '{tool_name}' 已执行。",
            "上一轮Observation摘要:",
            summary,
        ]
        if task_goal == "treatment" or self._is_treatment_request(self.patient_info):
            lines.append("")
            lines.append(self._build_treatment_readiness_context(cancer_type))
        lines.append("")
        lines.append("请基于摘要和agent状态决定下一步操作。不要复述长篇Observation；需要调用工具时只输出Thought/Action/Action Input。")
        return "\n".join(lines)

    #将完整文献结果设置到Python上下文变量中并调用治疗方案生成工具
    def _invoke_generate_treatment_plan_tool(self, tool_input: Dict[str, Any]):
        evidence_token = medical_tools.set_active_pubmed_evidence(self.last_literature_results)
        try:
            return TOOL_MAP["generate_treatment_plan"].invoke(tool_input)
        finally:
            medical_tools.reset_active_pubmed_evidence(evidence_token)

    #强制调用治疗方案生成工具
    def _force_generate_treatment_plan(self, round_number: int, thought: str, cancer_type: str) -> str:
        tool_input = self._fill_treatment_tool_input({}, cancer_type)
        missing_inputs = self._missing_treatment_tool_inputs(tool_input)
        if missing_inputs:
            observation = (
                "generate_treatment_plan工具返回失败，拒绝外层LLM直接输出治疗方案。"
                "当前缺少必要输入: " + "、".join(missing_inputs) + "。"
                "请重新分析当前目标、已有Observation和工具能力，选择下一步必要行动。"
            )
            self._record_step(round_number, thought, "final_rejected", {}, observation)
            return ""

        try:
            tool_result = self._invoke_generate_treatment_plan_tool(tool_input)
            self.last_rag_retrieved = medical_tools.LAST_TREATMENT_CONTEXT.get("rag_retrieved", self.last_rag_retrieved)
            self._remember_literature_results(medical_tools.LAST_TREATMENT_CONTEXT.get("literature_results", ""))
            self.last_treatment_plan = str(tool_result)
            self.final_answer = str(tool_result)
            self._record_step(round_number, thought, "generate_treatment_plan", tool_input, str(tool_result))
            return self.final_answer
        except Exception as e:
            error_msg = f"工具执行失败: {str(e)}"
            self._record_step(round_number, thought, "generate_treatment_plan", tool_input, error_msg)
            self.final_answer = (
                "已拒绝返回外层LLM自行生成的治疗方案，但治疗方案仍强制生成失败。\n"
                f"{error_msg}"
            )
            return self.final_answer

    #重置ReAct运行状态以处理下一个session
    def _reset_run_state(self, user_input: str, cancer_type: str, file_paths: Dict[str, str]):
        medical_tools.reset_active_pubmed_evidence()
        self.chat_history = []
        self.reasoning_steps = []
        self.tool_call_records = []
        self.last_literature_results = ""
        self.last_rag_retrieved = ""
        self.last_model_predictions = ""
        self.last_treatment_plan = ""
        self.final_answer = ""
        self.file_paths = file_paths
        self.patient_info = (
            f"用户输入: {user_input}\n"
            f"癌症类型: {cancer_type}\n"
            f"病理特征文件: {file_paths.get('features_input', '未提供')}\n"
            f"基因组数据文件: {file_paths.get('gene_input', '未提供')}"
        )

    #记录每一步的思考、行动、工具输入和观察结果，生成评估脚本
    def _record_step(self, round_number: int, thought: str, action: str, action_input: Any, observation: str):
        observation = self._format_observation_for_record(action, observation)
        step = {
            "round": round_number,
            "thought": thought,
            "action": action,
            "action_input": action_input if isinstance(action_input, dict) else str(action_input),
            "observation": observation,
        }
        self.reasoning_steps.append(step)
        self.tool_call_records.append({
            "round": round_number,
            "tool": action,
            "input": step["action_input"],
            "observation": observation,
        })

        formatted = (
            f"第{round_number}轮:\n"
            f"Thought: {thought}\n"
            f"Action: {action}\n"
            f"Action Input: {self._format_action_input_for_history(action, action_input)}\n"
            f"Observation: {observation}"
        )
        self.chat_history.append(formatted)
        return formatted
    def format_reasoning_chain(self, include_final: bool = True) -> str:
        lines = []
        for step in self.reasoning_steps:
            lines.append(
                f"第{step['round']}轮:\n"
                f"Thought: {step['thought']}\n"
                f"Action: {step['action']}\n"
                f"Action Input: {self._format_action_input_for_history(step['action'], step['action_input'])}\n"
                f"Observation: {step['observation']}"
            )
        if include_final and self.final_answer:
            lines.append(f"Final Answer: {self.final_answer}")
        return "\n\n".join(lines)
    def format_tool_calls(self) -> str:
        lines = []
        for item in self.tool_call_records:
            lines.append(
                f"{item['round']}. {item['tool']}("
                f"{self._format_action_input_for_history(item['tool'], item['input'])})"
            )
        return "\n".join(lines)
    def build_evaluation_record(self, name: str = None) -> Dict[str, Any]:
        return {
            "name": name or f"agent_case_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "patient_info": self.patient_info,
            "reasoning_chain": self.format_reasoning_chain(include_final=True),
            "tool_calls": self.format_tool_calls(),
            "model_predictions": self.last_model_predictions,
            "rag_retrieved": self.last_rag_retrieved,
            "literature_results": self.last_literature_results,
            "treatment_plan": self.final_answer,
        }
    def save_outputs(self, output_dir: str = None, cancer_type: str = "BRCA", base_name: str = None) -> Dict[str, str]:
        output_root = Path(output_dir or config.OUTPUT_DIR) / cancer_type
        output_root.mkdir(parents=True, exist_ok=True)
        safe_name = base_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_path = output_root / f"{safe_name}.txt"
        eval_path = output_root / f"{safe_name}.eval.json"

        txt_content = "\n".join([
            "=" * 60,
            "Agent 推理过程记录",
            "=" * 60,
            f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "【患者信息】",
            self.patient_info,
            "",
            "【ReAct推理过程】",
            self.format_reasoning_chain(include_final=False),
            "",
            "=" * 60,
            "【最终治疗方案】",
            "=" * 60,
            self.final_answer,
        ])

        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(txt_content)
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(self.build_evaluation_record(name=safe_name), f, ensure_ascii=False, indent=2)

        return {"txt": str(txt_path), "evaluation_json": str(eval_path)}
    
    def _parse_llm_response(self, response: str) -> Optional[tuple]:
        """解析LLM的响应，提取工具调用"""
        # 检查是否是Final Answer
        final_match = re.search(r'Final Answer:\s*(.+)$', response, re.DOTALL)
        if final_match:
            return ('final', final_match.group(1).strip())
        
        # 检查是否调用工具
        action_match = re.search(r'Action:\s*(\w+)', response)
        input_match = re.search(r'Action Input:\s*(\{[\s\S]*?\})', response)
        
        if action_match and input_match:
            tool_name = action_match.group(1).strip()
            try:
                # 尝试解析JSON
                tool_input = json.loads(input_match.group(1))
                return ('tool', tool_name, tool_input)
            except:
                # 如果JSON解析失败，尝试简单解析
                return ('tool', tool_name, input_match.group(1))
        
        return None
    
    def run(self, user_input: str, cancer_type: str = "BRCA", task_goal: str | None = None) -> str:
        """运行医学任务ReAct流程"""
        
        if self.verbose:
            print("\n" + "="*60)
            print("医学任务ReAct Agent")
            print("="*60)
        
        # 提取文件路径
        file_paths = self._extract_file_paths(user_input)
        if self.verbose and file_paths:
            print(f"提取到文件: {file_paths}")
        
        # 初始化运行状态
        self._reset_run_state(user_input, cancer_type, file_paths)
        requires_treatment_plan = task_goal == "treatment" or (
            task_goal is None and self._is_treatment_request(user_input)
        )
        
        # 构建初始消息
        context = f"""用户输入: {user_input}

        可用文件路径:
        - features_input: {file_paths.get('features_input', '无')}
        - gene_input: {file_paths.get('gene_input', '无')}
        - cancer_type: {cancer_type}
        - 当前医学任务边界: {self._task_goal_label(task_goal)}

        你当前只负责当前病例相关医学任务，并必须通过ReAct循环自主选择工具：
        - 先分析用户目标、已有患者信息和可用文件路径。
        - 自主判断下一步需要哪个工具；不要按固定任务模板机械调用。
        - 只调用对当前目标必要的工具；如果缺少必要文件或证据，说明缺少什么。
        - 不要扩展到当前医学任务边界之外的任务。
        - 当某个工具Observation已经满足当前医学任务边界时，应该结束当前ReAct任务。
        - 如果最终涉及治疗方案，必须先具备T/N/M、临床分期、预测生存时间、风险等级和有效PubMed检索证据，不能自造患者状态或文献证据。

        请按医学任务ReAct流程一步步执行。
        """

        # ReAct循环
        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n--- 第{iteration + 1}轮 ---")
            
            # 构建消息
            messages = [SystemMessage(content=REACT_SYSTEM_PROMPT)]
            
            # 添加历史
            for msg in self.chat_history:
                messages.append(HumanMessage(content=msg))
            
            # 添加当前上下文
            messages.append(HumanMessage(content=context))
            
            # 调用LLM
            response = _invoke_llm_with_retry(
                self.llm,
                messages,
                call_name=f"ReAct第{iteration + 1}轮",
            )
            llm_response = response.content if hasattr(response, 'content') else str(response)
            
            if self.verbose:
                print(f"LLM响应: \n{llm_response[:2500]}...\n")
            
            # 解析响应
            parsed = self._parse_llm_response(llm_response)
            
            if parsed is None:
                # 无法解析，尝试直接回复
                observation = "LLM响应无法解析为标准ReAct格式，请继续使用Thought/Action/Action Input格式。"
                self._record_step(iteration + 1, self._extract_thought(llm_response), "parse_error", {}, observation)
                context = "请继续使用工具完成诊断，或给出最终回复。"
                continue
            
            if parsed[0] == 'final':
                # 完成
                if requires_treatment_plan and not self.last_treatment_plan:
                    forced_answer = self._force_generate_treatment_plan(
                        iteration + 1,
                        "外层LLM试图在未成功调用generate_treatment_plan前直接给出治疗方案；系统强制改为工具调用。",
                        cancer_type,
                    )
                    if forced_answer:
                        if self.verbose:
                            print("\n" + "="*60)
                            print("诊断完成")
                            print("="*60)
                        return forced_answer
                    context = (
                        "当前尚未获得generate_treatment_plan工具结果，不能让外层推理LLM直接得到Final Answer。"
                        "请重新分析当前目标、已有Observation和工具前置条件，继续选择必要工具。"
                        "如果最终涉及治疗方案，必须使用具备CSCO和PubMed证据约束的工具结果作为最终答案。"
                    )
                    continue

                if self.last_treatment_plan:
                    self.final_answer = self.last_treatment_plan
                else:
                    self.final_answer = parsed[1]
                if self.verbose:
                    print("\n" + "="*60)
                    print("诊断完成")
                    print("="*60)
                return self.final_answer
            
            # 调用工具
            tool_name = parsed[1]
            tool_input = parsed[2] if len(parsed) > 2 else {}
            
            if tool_name not in TOOL_MAP:
                context = f"错误: 未知工具 '{tool_name}'。请使用有效的工具名。"
                self._record_step(iteration + 1, self._extract_thought(llm_response), tool_name, tool_input, context)
                continue

            if not isinstance(tool_input, dict):
                context = "错误: Action Input必须是JSON对象。"
                self._record_step(iteration + 1, self._extract_thought(llm_response), tool_name, tool_input, context)
                continue
            
            # 补充参数
            tool_input.setdefault('cancer_type', cancer_type)
            if tool_name == 'search_literature':
                tool_input.setdefault('max_results', 8)
            # 保证文件路径正确
            if tool_name == 'predict_tnm_stage' :
                if 'features_input' not in file_paths:
                    context = "错误: 调用predict_tnm_stage前缺少features_input(病理特征文件路径)。请先提供病理特征文件。"
                    print(context)
                    self._record_step(iteration + 1, self._extract_thought(llm_response), tool_name, tool_input, context)
                    continue
                tool_input["features_input"] = file_paths['features_input']
            if tool_name == 'predict_survival':
                missing_paths = []
                if 'features_input' not in file_paths:
                    missing_paths.append("features_input(病理特征文件路径)")
                if 'gene_input' not in file_paths:
                    missing_paths.append("gene_input(基因组数据CSV路径)")
                if missing_paths:
                    context = f"错误: 调用predict_survival前缺少必要文件路径: {', '.join(missing_paths)}。"
                    print(context)
                    self._record_step(iteration + 1, self._extract_thought(llm_response), tool_name, tool_input, context)
                    continue
                tool_input["features_input"] = file_paths['features_input']
                tool_input["gene_input"] = file_paths['gene_input']
            if tool_name == 'generate_treatment_plan':
                tool_input = self._fill_treatment_tool_input(tool_input, cancer_type)
                missing_inputs = self._missing_treatment_tool_inputs(tool_input)
                if missing_inputs:
                    context = (
                        "错误: generate_treatment_plan缺少必要输入，当前不能执行；"
                        "缺少: " + "、".join(missing_inputs) + "。"
                        "请重新分析当前目标、已有Observation和工具能力，选择下一步必要行动。"
                    )
                    print(context)
                    self._record_step(iteration + 1, self._extract_thought(llm_response), tool_name, tool_input, context)
                    continue
            
            if self.verbose:
                print(f"调用工具: {tool_name}")
                print(f"参数: {tool_input}\n")
                print("工具分析流程：")
            
            # 执行工具
            try:
                if tool_name == "generate_treatment_plan":
                    tool_result = self._invoke_generate_treatment_plan_tool(tool_input)
                else:
                    tool_func = TOOL_MAP[tool_name]
                    tool_result = tool_func.invoke(tool_input)
                 
                if self.verbose:
                    print(f"工具结果:\n {str(tool_result)[:2500]}...")
                
                if tool_name == "search_literature":
                    self._remember_literature_results(tool_result)
                elif tool_name in {"predict_tnm_stage", "predict_survival"}:
                    self.last_model_predictions = (
                        f"{self.last_model_predictions}\n\n[{tool_name}]\n{tool_result}"
                    ).strip()
                elif tool_name == "generate_treatment_plan":
                    self.last_rag_retrieved = medical_tools.LAST_TREATMENT_CONTEXT.get("rag_retrieved", self.last_rag_retrieved)
                    self._remember_literature_results(medical_tools.LAST_TREATMENT_CONTEXT.get("literature_results", ""))
                    self.last_treatment_plan = str(tool_result)

                # 记录到历史
                self._record_step(iteration + 1, self._extract_thought(llm_response), tool_name, tool_input, str(tool_result))

                if self._tool_satisfies_task_goal(task_goal, tool_name):
                    self.final_answer = str(tool_result)
                    if self.verbose:
                        print("\n" + "="*60)
                        print("诊断完成")
                        print("="*60)
                    return self.final_answer

                if tool_name == "generate_treatment_plan":
                    self.final_answer = str(tool_result)
                    if self.verbose:
                        print("\n" + "="*60)
                        print("诊断完成")
                        print("="*60)
                    return self.final_answer
                
                # 更新给下一轮LLM的上下文时只传摘要，完整工具结果保存在agent状态中。
                context = self._build_followup_context(
                    tool_name,
                    str(tool_result),
                    cancer_type,
                    task_goal,
                )
                
            except Exception as e:
                error_msg = f"工具执行失败: {str(e)}"
                if self.verbose:
                    print(f"错误: {error_msg}")
                
                self._record_step(iteration + 1, self._extract_thought(llm_response), tool_name, tool_input, error_msg)
                if tool_name == "generate_treatment_plan":
                    self.final_answer = (
                        "治疗方案生成失败，已拒绝返回外层LLM自行生成的治疗方案。\n"
                        f"{error_msg}"
                    )
                    return self.final_answer
                else:
                    context = f"工具执行失败: {error_msg}。请尝试其他医学工具或说明失败原因。"
        
        # 超过最大迭代次数
        if requires_treatment_plan and not self.last_treatment_plan:
            self.final_answer = (
                "治疗方案生成失败：已达到最大迭代次数，但没有成功执行generate_treatment_plan工具。"
                "已拒绝返回外层LLM自行生成的治疗方案。"
            )
        else:
            self.final_answer = "已达到最大迭代次数，请检查输入或简化问题。"
        return self.final_answer


def create_autonomous_agent(verbose: bool = True, llm_model: str | None = None) -> MedicalTaskReActAgent:
    """创建医学任务ReAct Agent。"""
    return MedicalTaskReActAgent(verbose=verbose, llm_model=llm_model)


# ==================== Demo入口 ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='医学任务ReAct智能体')
    parser.add_argument('--features_input', type=str, default=None)
    parser.add_argument('--gene_input', type=str, default=None)
    parser.add_argument('--cancer_type', type=str, default='BRCA')
    parser.add_argument('--query', type=str, default=None)
    parser.add_argument('--output', type=str, default=str(config.OUTPUT_DIR))
    
    args = parser.parse_args()
    
    # 构建输入
    if args.query:
        user_input = args.query
    else:
        parts = [f"请分析癌症类型为{args.cancer_type}的患者"]
        if args.features_input:
            parts.append(f"病理特征文件: {args.features_input}")
        if args.gene_input:
            parts.append(f"基因组数据文件: {args.gene_input}")
        user_input = "，".join(parts) + "。请生成治疗方案。"
    
    # 运行
    agent = create_autonomous_agent(verbose=True)
    result = agent.run(user_input, cancer_type=args.cancer_type)
    
    print("\n" + "="*60)
    print("最终医学任务结果")
    print("="*60)
    print(result)

    paths = agent.save_outputs(args.output, cancer_type=args.cancer_type)
    print(f"\n推理过程TXT已保存: {paths['txt']}")
    print(f"评估输入JSON已保存: {paths['evaluation_json']}")


# Backward-compatible public names.
AutonomousReActAgent = MedicalTaskReActAgent
TreatmentPlanReActAgent = MedicalTaskReActAgent
