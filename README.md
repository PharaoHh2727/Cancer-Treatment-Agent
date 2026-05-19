# 简介

一个本地自托管的医学多模态智能体工作台。前端使用 React，后端使用 FastAPI，后端在使用者自己的机器上运行 CLAM、TransMIL、CMTA、CSCO RAG、PubMed 检索和智谱 AI Agent。

## 技术栈

```text
前端：React
后端：FastAPI
后端部署的医学模型和大模型：CLAM / TransMIL / CMTA / RAG / PubMed / ZhipuAI
```

默认情况下，用户前端上传的 WSI、CSV、CLAM 输出和各医学模型中间结果都保存在本地 `runtime_data/`。

## 目录结构

```text
CLAM-master/          CLAM 代码
backend/              FastAPI 后端
chroma_db/            RAG 向量库（运行rag.py时自动生成，无需自己上传）
cmta_survival/        CMTA模型代码与权重目录
frontend/             React 前端
gene_data/            CMTA中六组OMIC基因列表
pubmed/               PubMed文献检索
transmil_tnm/         TransMIL模型代码与权重目录
CSCO/                 原始 CSCO 指南 PDF 文件（需自己新建文件夹并上传文件）
autonomous_agent.py   医学 ReAct 智能体
config.py             路径和模型配置
demo.py               python demo.py --features_input 病理pt特征文件路径 --gene_input 基因组数据文件路径
                      --cancer_type 癌种（运行单个病例并生成治疗方案和评估脚本）
medical_tools.py      医学工具
rag.py                CSCO RAG
.env.example          环境变量模板
docker-compose.yml    一键本地部署
```

## 环境变量

复制环境变量模板：

```text
cp .env.example .env
```

填写API KEY：

```text
ZHIPUAI_API_KEY=你的智谱AI密钥
PUBMED_API_KEY=你的PubMed API Key
```

`PUBMED_API_KEY` 可选，不填也可以低频检索。`ZHIPUAI_API_KEY` 对需要 LLM 的 Agent 推理是必需的。

## 1、Docker 启动

推荐开源用户使用 Docker Compose：

```text
docker compose up --build
```

打开：

```text
http://127.0.0.1:5173
```

后端 API：

```text
http://127.0.0.1:8000/api/health
http://127.0.0.1:8000/docs
```

## 2、手动启动

后端：

手动安装时需要系统已安装 OpenSlide。Linux 可使用 `apt install openslide-tools libopenslide0`，Windows 建议优先使用 Docker，或自行安装 OpenSlide Windows 运行库并加入 `PATH`。

后端：

```text
cd 项目目录/backend
先安装后端依赖：pip install -r requirements.txt
运行后端代码：python -m backend.api_server
```

前端：

```text
cd 项目目录/frontend
npm install
npm run dev
打开前端网页：http://127.0.0.1:5173
```

## 模型和数据文件

开源发布时，如果模型权重、CSCO PDF文件过大，建议使用 Git LFS 或单独下载链接分发。默认路径如下：

```text
transmil_tnm/results/         研究癌种的TransMIL模型权重
cmta_survival/results/        研究癌种的CMTA模型权重
CSCO/                         研究癌种的CSCO PDF文件
gene_data/                    CMTA中六组OMIC基因列表
```

也可以在 `.env` 中修改路径：

```bash
TRANSMIL_MODEL_DIR=./transmil_tnm/results
CMTA_RESULTS_DIR=./cmta_survival/results
GUIDE_PDF_DIR=./CSCO
GENE_DATA_DIR=./gene_data
```

## 后端主要接口

```text
GET  /api/health
POST /api/upload/pathology
POST /api/upload/genome
POST /api/clam/run
POST /api/agent/chat
GET  /api/files/{session_id}/{relative_path}
```

## 隐私说明

本项目默认是本地自托管模式。用户上传的数据进入自己机器上的 FastAPI 后端和 `runtime_data/`，不依赖项目作者服务器。若部署到机构服务器，需要由部署方自行配置访问控制、HTTPS、审计日志和数据保留策略。
