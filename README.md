# Medical Agent Workbench

一个本地自托管的医学多模态智能体工作台。前端使用 React，后端使用 FastAPI，后端在使用者自己的机器上运行 CLAM、TransMIL、CMTA、CSCO RAG、PubMed 检索和智谱 AI Agent。

## 架构

```text
browser
  -> React frontend
  -> FastAPI backend
  -> CLAM / TransMIL / CMTA / RAG / PubMed / ZhipuAI
```

默认情况下，用户上传的 WSI、CSV、CLAM 输出和模型中间结果都保存在本地 `runtime_data/`，不会发送到项目作者服务器。

## 目录结构

```text
agent/
  backend/              FastAPI 后端
  frontend/             React 前端
  CLAM-master/          CLAM 代码
  transmil_tnm/         TNM 模型代码与权重目录
  cmta_survival/        生存预测模型代码与权重目录
  CSCO/                 CSCO 指南 PDF
  chroma_db/            RAG 向量库
  autonomous_agent.py   自主 ReAct 智能体
  config.py             路径和模型配置
  .env.example          环境变量模板
  docker-compose.yml    一键本地部署
```

## 环境变量

先复制环境变量模板：

```bash
cp .env.example .env
```

至少填写：

```bash
ZHIPUAI_API_KEY=你的智谱AI密钥
PUBMED_API_KEY=你的PubMed API Key
```

`PUBMED_API_KEY` 可选，不填也可以低频检索。`ZHIPUAI_API_KEY` 对需要 LLM 的 Agent 推理是必需的。

## Docker 启动

推荐开源用户使用 Docker Compose：

```bash
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

## 手动启动

后端：

手动安装时需要系统已安装 OpenSlide。Linux 可使用 `apt install openslide-tools libopenslide0`，Windows 建议优先使用 Docker，或自行安装 OpenSlide Windows 运行库并加入 `PATH`。

```bash
cd agent
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
python -m backend.api_server
```

Windows PowerShell：

```powershell
cd agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
python -m backend.api_server
```

前端：

```bash
cd frontend
npm install
npm run dev
```

打开：

```text
http://127.0.0.1:5173
```

## 模型和数据文件

开源发布时，如果模型权重、CSCO PDF 或 Chroma 向量库文件过大，建议使用 Git LFS 或单独下载链接分发。默认路径如下：

```text
transmil_tnm/results/
cmta_survival/results/
CSCO/
chroma_db/
gene_data/
```

也可以在 `.env` 中覆盖：

```bash
TRANSMIL_MODEL_DIR=/path/to/transmil_tnm/results
CMTA_RESULTS_DIR=/path/to/cmta_survival/results
GUIDE_PDF_DIR=/path/to/CSCO
RAG_DB_DIR=/path/to/chroma_db
GENE_DATA_DIR=/path/to/gene_data
```

## 主要接口

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
