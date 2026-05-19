from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import HumanMessage, SystemMessage


ToolHandler = Callable[[dict[str, Any]], Any]


@dataclass
class GeneralTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _strip_json_fence(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _parse_control_message(text: str) -> dict[str, Any] | None:
    cleaned = _strip_json_fence(text)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _tool_catalog(tools: list[GeneralTool]) -> str:
    return "\n".join(
        [
            (
                f"- {tool.name}: {tool.description}\n"
                f"  input_schema: {_json_dumps(tool.input_schema)}"
            )
            for tool in tools
        ]
    )


def _beijing_time() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S %Z")


def _general_model_name(fallback_model: str) -> str:
    return fallback_model.strip() or "glm-4-flash"


def _invoke_zhipu(messages: list[dict[str, str]], model_name: str) -> str:
    llm = ChatZhipuAI(
        model=model_name,
        temperature=0.3,
    )
    lc_messages = []
    for message in messages:
        if message["role"] == "system":
            lc_messages.append(SystemMessage(content=message["content"]))
        else:
            lc_messages.append(HumanMessage(content=message["content"]))
    response = llm.invoke(lc_messages)
    return response.content if hasattr(response, "content") else str(response)


def _invoke_general_llm(messages: list[dict[str, str]], fallback_model: str) -> tuple[str, str]:
    model_name = _general_model_name(fallback_model)
    return _invoke_zhipu(messages, model_name), model_name


def _build_system_prompt(
    tools: list[GeneralTool],
    runtime_context: dict[str, Any],
    fallback_model: str,
) -> str:
    general_model = _general_model_name(fallback_model)
    enriched_context = {
        **runtime_context,
        "current_beijing_time": _beijing_time(),
        "general_assistant_model": general_model,
        "medical_react_model": fallback_model,
    }
    return f"""你是系统的唯一对话入口，也是主智能体。

            你可以直接回答日常问答、系统说明、时间、能力介绍、普通知识和非医学问题。
            当用户的问题需要本地医学专用能力时，你可以调用一个工具。工具执行后，你会收到 Observation，然后再给用户最终回答。

            重要边界:
            - 不要把所有问题都交给工具。普通聊天、时间、模型信息、工具列表、项目说明、非患者医学科普可以直接回答。
            - 只有当用户需要基于本地患者数据进行 TNM 分期预测、生存时间/风险预测，或生成个体化治疗方案时，才调用医学工具。
            - 如果运行时上下文显示 pathology_pt_ready=true 且用户需要当前病例分期预测，已有数据足够进入医学ReAct分期入口，不要追问年龄、性别或其它通用信息。
            - 如果运行时上下文显示 pathology_pt_ready=true 且 genome_csv_uploaded=true 且用户需要当前病例生存时间/风险预测，已有数据足够进入医学ReAct生存入口，不要追问预测年限、年龄、性别或其它通用信息。
            - 如果运行时上下文显示当前病例数据已上传且用户需要当前病例治疗方案，进入医学ReAct治疗方案入口；缺少的分期、预后和文献证据由医学ReAct内部继续处理。
            - 你不能自己编造本地模型预测结果、CSCO RAG 证据或 PubMed 检索证据。
            - 如果工具返回缺少文件或信息，你要直接告诉用户缺什么，以及下一步应该上传或处理什么。
            - 回答医学结果时，保持审慎，说明这是模型辅助分析，不能替代临床医生判断。
            - 对工具返回的分期、风险等级、生存时间、热力图路径、CSCO页码和PMID，必须原样保留关键值，不要自行修正或编造。

            运行时上下文:
            {_json_dumps(enriched_context)}

            可调用工具:
            {_tool_catalog(tools)}

            你每次必须只输出一个 JSON 对象，不要输出 Markdown 代码块。

            直接回答时:
            {{"type":"final","answer":"给用户的自然语言回答"}}

            需要调用工具时:
            {{"type":"tool","tool":"工具名","arguments":{{"参数名":"参数值"}}}}
        """


def _looks_like_missed_medical_tool(answer: str, runtime_context: dict[str, Any]) -> bool:
    text = answer or ""
    available = runtime_context.get("available_patient_data") or {}
    has_any_local_data = bool(
        available.get("pathology_pt_ready")
        or available.get("genome_csv_uploaded")
        or available.get("pathology_wsi_uploaded")
    )
    if not has_any_local_data:
        return False
    tool_need_markers = [
        "需要使用医学工具",
        "调用正确的工具",
        "进行预测",
        "进行分析",
    ]
    ask_again_markers = [
        "请提供",
        "需要提供",
        "相关信息",
        "以便",
    ]
    return any(marker in text for marker in tool_need_markers) and any(
        marker in text for marker in ask_again_markers
    )


def _extract_embedded_tool_request(answer: str, tool_names: set[str]) -> dict[str, Any] | None:
    text = answer or ""
    parsed = _parse_control_message(text)
    if parsed and str(parsed.get("type") or "").lower() == "tool":
        tool_name = str(parsed.get("tool") or "").strip()
        if tool_name in tool_names:
            arguments = parsed.get("arguments") or {}
            return {
                "type": "tool",
                "tool": tool_name,
                "arguments": arguments if isinstance(arguments, dict) else {},
            }

    for tool_name in tool_names:
        if tool_name not in text:
            continue
        request_match = re.search(r'"request"\s*:\s*"([^"]*)"', text)
        cancer_match = re.search(r'"cancer_type"\s*:\s*"([^"]*)"', text)
        arguments: dict[str, Any] = {}
        if request_match:
            arguments["request"] = request_match.group(1)
        if cancer_match:
            arguments["cancer_type"] = cancer_match.group(1)
        return {
            "type": "tool",
            "tool": tool_name,
            "arguments": arguments,
        }

    return None


def run_general_assistant(
    query: str,
    tools: list[GeneralTool],
    runtime_context: dict[str, Any],
    llm_model: str,
    max_iterations: int = 4,
) -> dict[str, Any]:
    tool_map = {tool.name: tool for tool in tools}
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": _build_system_prompt(tools, runtime_context, llm_model),
        },
        {"role": "user", "content": query},
    ]
    tool_calls: list[dict[str, Any]] = []
    model_name = _general_model_name(llm_model)

    for _ in range(max_iterations):
        raw_response, model_name = _invoke_general_llm(messages, llm_model)
        control = _parse_control_message(raw_response)
        if not control:
            return {
                "answer": raw_response.strip(),
                "route": "general",
                "general_model": model_name,
                "tool_calls": tool_calls,
            }

        response_type = str(control.get("type") or "").lower()
        if response_type == "final":
            answer = str(control.get("answer") or "").strip()
            embedded_tool = _extract_embedded_tool_request(answer, set(tool_map))
            if embedded_tool:
                control = embedded_tool
                response_type = "tool"
            elif not tool_calls and _looks_like_missed_medical_tool(answer, runtime_context):
                messages.append({"role": "assistant", "content": raw_response})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "你刚才判断需要使用医学工具，但没有发起工具调用。"
                            "运行时上下文已经提供了当前病例的上传数据状态。"
                            "请重新分析可用工具和数据：如果某个医学入口适合当前用户目标，输出tool JSON；"
                            "只有在确实没有合适工具或必要文件缺失时，才输出final JSON说明具体缺失项。"
                        ),
                    }
                )
                continue
            else:
                return {
                    "answer": answer,
                    "route": tool_calls[-1]["tool"] if tool_calls else "general",
                    "general_model": model_name,
                    "tool_calls": tool_calls,
                }

        if response_type != "tool":
            return {
                "answer": raw_response.strip(),
                "route": "general",
                "general_model": model_name,
                "tool_calls": tool_calls,
            }

        tool_name = str(control.get("tool") or "").strip()
        arguments = control.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}
        if tool_name not in tool_map:
            observation = f"未知工具: {tool_name}。可用工具: {', '.join(tool_map)}"
        else:
            try:
                observation = tool_map[tool_name].handler(arguments)
            except Exception as exc:
                observation = f"工具 {tool_name} 执行失败: {exc}"

        observation_text = observation if isinstance(observation, str) else _json_dumps(observation)
        tool_calls.append(
            {
                "tool": tool_name,
                "arguments": arguments,
                "observation": observation_text,
            }
        )

        if tool_name in {"predict_tnm_stage", "predict_survival", "medical_treatment_plan"}:
            return {
                "answer": observation_text,
                "route": tool_name,
                "general_model": model_name,
                "tool_calls": tool_calls,
            }

        messages.append({"role": "assistant", "content": raw_response})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Observation from {tool_name}:\n{observation_text}\n\n"
                    "请基于这个 Observation 继续。若已经足够回答用户，请输出 final JSON；"
                    "若还必须调用另一个工具，请输出 tool JSON。"
                ),
            }
        )

    return {
        "answer": "主智能体已达到最大工具调用轮数，请简化问题或分步提问。",
        "route": tool_calls[-1]["tool"] if tool_calls else "general",
        "general_model": model_name,
        "tool_calls": tool_calls,
    }
