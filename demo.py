"""
演示Demo - 自主决策医学智能体
"""
import os
import sys
from pathlib import Path
from datetime import datetime

# 设置环境
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 添加项目根目录
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

import config
from autonomous_agent import create_autonomous_agent


def save_result_to_file(agent, final_result: str, output_path: str):
    """将推理过程和最终结果保存到文件"""
    output = []
    output.append("=" * 60)
    output.append("Agent 推理过程记录")
    output.append("=" * 60)
    output.append(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append("")

    output.append("【推理过程】")
    output.append("-" * 40)
    output.append(agent.format_reasoning_chain(include_final=False))

    output.append("\n")
    output.append("=" * 60)
    output.append("【最终治疗方案】")
    output.append("=" * 60)
    output.append(final_result)

    content = "\n".join(output)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\n结果已保存到: {output_path}")


def run_demo(
    features_input: str = None,
    gene_input: str = None,
    cancer_type: str = "BRCA",
    query: str = None,
    output: str = None
):
    """运行自主决策医学智能体"""
    print("=" * 60)
    print("自主决策医学智能体")
    print("=" * 60)
    print(f"癌症类型: {cancer_type}")
    print(f"病理特征: {features_input}")
    print(f"基因组数据: {gene_input}")

    # 构建输入
    if query is None:
        parts = [f"请分析癌症类型为{cancer_type}的患者"]

        if features_input:
            parts.append(f"病理特征文件: {features_input}")

        if gene_input:
            parts.append(f"基因组数据文件: {gene_input}")

        query = "，".join(parts) + "。请生成治疗方案。"

    print(f"\n查询: {query}")
    print("-" * 60)

    # 创建Agent并运行
    agent = create_autonomous_agent(verbose=True)
    result = agent.run(query, cancer_type)

    print("\n" + "=" * 60)
    print("分析结果")
    print("=" * 60)
    print(result)

    # 保存结果到文件，同时生成评估脚本可直接读取的JSON
    saved = agent.save_outputs(output or config.OUTPUT_DIR, cancer_type=cancer_type)
    print(f"推理过程TXT: {saved['txt']}")
    print(f"评估输入JSON: {saved['evaluation_json']}")
    

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='自主决策医学智能体')
    parser.add_argument('--features_input', type=str, default=None,
                        help='病理特征文件路径 (.pt/.npy)')
    parser.add_argument('--gene_input', type=str, default=None,
                        help='基因组数据文件路径 (.csv/.npy)')
    parser.add_argument('--cancer_type', type=str, default='BRCA',
                        help='癌症类型 (BRCA/LUAD/BLCA)')
    parser.add_argument('--query', type=str, default=None,
                        help='自定义查询文本')
    parser.add_argument('--output', type=str, default=None,
                        help='输出文件路径')

    args = parser.parse_args()

    # 检查是否设置了API Key
    pubmed_key = os.getenv('PUBMED_API_KEY', '')
    if not pubmed_key:
        print("\n提示: 未设置PUBMED_API_KEY环境变量，PubMed仍可低频检索；设置API Key可提高请求额度。")
        print("设置方式: set PUBMED_API_KEY=你的API_KEY\n")

    # 运行
    run_demo(
        features_input=args.features_input,
        gene_input=args.gene_input,
        cancer_type=args.cancer_type,
        query=args.query,
        output=config.OUTPUT_DIR
    )
