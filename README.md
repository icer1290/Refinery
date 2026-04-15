[中文](README.md) | [English](README.en.md)

# Refinery

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

基于 LangGraph 构建的 AI 科技新闻聚合系统，采用 CLI 优先设计，支持智能体无缝集成。

## 功能特性

- **RSS 聚合**: 自动从 17+ 科技新闻源抓取新闻
- **语义去重**: 使用向量嵌入识别并去除重复文章
- **多维评分**: 基于「行业影响」「里程碑意义」「关注度」的 AI 评分系统
- **内容提取**: 使用 trafilatura 提取完整文章内容
- **中文翻译**: 生成中文标题与摘要，保留实体信息
- **自我反思**: 验证翻译质量，自动重试机制
- **深度搜索**: 基于 ReAct 循环的按需深度研究（支持 DuckDuckGo/Tavily 网络搜索）
- **GraphRAG**: 知识图谱构建与社区发现，支持上下文分析
- **多轮对话**: 多智能体架构的对话式 AI，支持 ReAct 循环与多层记忆
- **向量存储**: PostgreSQL + pgvector 扩展，支持向量存储与图数据
- **CLI 接口**: 完整命令行工具，支持 JSON 输出，便于智能体自动化

## 安装

### 从 GitHub 安装

```bash
# 最新版本
pip install git+https://github.com/icer1290/Refinery.git

# 指定版本/标签
pip install git+https://github.com/icer1290/Refinery.git@v1.0.0
```

### 从源码安装（开发模式）

```bash
git clone https://github.com/icer1290/Refinery.git
cd Refinery

# 使用 uv（推荐）
uv sync

# 或使用 pip
pip install -e .
```

### 环境要求

- Python 3.11+
- Docker & Docker Compose（用于 PostgreSQL 和 Redis）
- LLM API Key（OpenAI、DashScope、Azure 或兼容服务）

## 快速开始

```bash
# 1. 初始化配置
refinery init

# 2. 启动 Docker 服务（PostgreSQL、Redis）
refinery services start

# 3. 运行新闻聚合工作流
refinery workflow run

# 4. 查看已收集的文章
refinery article list
```

## CLI 命令

### 服务管理

```bash
# 启动所有服务
refinery services start

# 启动指定服务
refinery services start postgres redis

# 查看服务状态
refinery services status

# 查看日志
refinery services logs postgres -f

# 停止服务
refinery services stop

# 检查是否运行（适用于脚本）
refinery services is-running
```

### 工作流命令

```bash
# 运行工作流（默认：最近 24 小时）
refinery workflow run

# 使用指定 RSS 源运行
refinery workflow run -f https://feeds.arstechnica.com/arstechnica/technology-lab

# 设置评分阈值
refinery workflow run --threshold 6.0

# 强制重新处理已存在的文章
refinery workflow run --force

# JSON 输出（智能体友好）
refinery workflow run --json

# 查看工作流运行历史
refinery workflow list

# 查看指定运行详情
refinery workflow show <run-id>
```

### 文章命令

```bash
# 列出文章
refinery article list

# 分页查询
refinery article list --page 2 --size 50

# 按最低评分筛选
refinery article list --min-score 7.0

# 按来源筛选
refinery article list --source "TechCrunch"

# JSON 输出
refinery article list --json

# 查看文章详情
refinery article show <article-id>
```

### 深度搜索命令

```bash
# 运行深度搜索
refinery search run <article-id>

# 设置最大迭代次数
refinery search run <article-id> --iterations 10

# JSON 输出
refinery search run <article-id> --json

# 检查是否已执行深度搜索
refinery search status <article-id>
```

### 图谱命令

```bash
# 构建知识图谱
refinery graph build <article-id-1> <article-id-2>

# 分析图谱
refinery graph analyze <article-id>

# 自定义扩展设置
refinery graph analyze <article-id> --hops 3 --expansion 100
```

### 对话命令

```bash
# 启动交互式对话
refinery chat <article-id>

# 指定用户 ID
refinery chat <article-id> --user 42
```

## JSON 输出

所有命令支持 `--json` 标志输出结构化数据，便于脚本编写和智能体集成：

```bash
refinery article list --json
```

输出示例：
```json
{
  "articles": [
    {
      "id": "abc123",
      "source_name": "TechCrunch",
      "chinese_title": "AI技术突破",
      "total_score": 8.5,
      "published_at": "2024-01-15 10:30"
    }
  ],
  "total": 42,
  "page": 1,
  "size": 20
}
```

## 配置

### 配置优先级

| 优先级 | 来源 | 说明 |
|--------|------|------|
| 1（最高） | CLI 参数 | `--option` 标志 |
| 2 | 环境变量 | `OPENAI_API_KEY` 等 |
| 3 | 本地配置 | `./.refinery.toml` |
| 4 | 全局配置 | `~/.refinery/config.toml` |
| 5（最低） | 默认值 | 内置默认配置 |

### 配置向导

运行交互式配置向导：

```bash
refinery init
```

### 环境变量

| 变量 | 必填 | 说明 |
|------|------|------|
| `DATABASE_URL` | 是 | PostgreSQL 连接 URL |
| `OPENAI_API_KEY` | 是 | OpenAI 或兼容 API Key |
| `OPENAI_CHAT_MODEL` | 是 | 聊天模型（如 gpt-4o-mini、qwen3.5-35b-a3b） |
| `OPENAI_EMBEDDING_MODEL` | 是 | 嵌入模型 |
| `OPENAI_BASE_URL` | 否 | 自定义 API 地址（用于 DashScope、Azure、Ollama） |
| `WEB_SEARCH_PROVIDER` | 否 | 搜索引擎：`duckduckgo` 或 `tavily`（默认 duckduckgo） |
| `REDIS_URL` | 否 | Redis 连接 URL |
| `REDIS_ENABLED` | 否 | 启用 Redis 缓存（默认 true） |

完整配置选项请参阅 `.env.example`。

## 系统架构

### 主工作流管道

```
[Entry] → [Scout] → [Dedup] → [Scoring] → [Writing] → [Reflection] → [Storage] → [End]
```

- **Scout**: 抓取 RSS 源，提取文章
- **Dedup**: 向量相似度去重
- **Scoring**: 多维 AI 评分（行业影响、里程碑、关注度）
- **Writing**: 内容提取 + 中文翻译
- **Reflection**: 翻译质量验证与重试
- **Storage**: 持久化到 PostgreSQL 并存储向量

### 深度搜索 (ReAct 循环)

按需深度研究流程：
1. 抓取文章内容
2. ReAct 循环配合网络搜索工具
3. 生成综合报告，存储在 `deepsearch_report` 字段

### GraphRAG (DeepGraph)

双阶段知识图谱系统：
- **后台 GraphBuilder**: 提取实体/关系，社区发现（Leiden 算法）
- **按需 GraphAnalyst**: 获取子图，图遍历扩展，生成分析

### 多轮对话 (Hub-and-Spoke 架构)

多智能体协作的对话式 AI：
- **Supervisor**: 中央调度器，评估意图并路由至专家智能体
- **Researcher**: ReAct 循环智能体，深度信息收集
- **Explainer**: 提供文章解释与上下文
- **Fact Checker**: 对照知识图谱与网络来源验证观点
- **多层记忆**: 短期、中期、长期三层记忆系统

## 技术栈

- **后端**: FastAPI, LangGraph, LangChain
- **数据库**: PostgreSQL + pgvector
- **缓存**: Redis（可选）
- **AI**: OpenAI 或兼容 API
- **RSS/Web**: feedparser, trafilatura, duckduckgo-search
- **CLI**: Typer, Rich

## API 接口

所有接口位于 `/api/v1/` 路径下：

| 接口 | 方法 | 说明 |
|------|------|------|
| `/workflow/run` | POST | 触发新闻聚合工作流 |
| `/workflow/runs` | GET | 查询工作流运行历史 |
| `/workflow/articles` | GET | 获取文章列表 |
| `/workflow/articles/{id}` | GET | 获取文章详情 |
| `/deep-search/run` | POST | 执行深度搜索 |
| `/deep-graph/analyze` | POST | 生成知识图谱分析 |
| `/chat/chat` | POST | 发送消息并获取 AI 回复 |

API 文档地址：http://localhost:8000/docs

## Docker 部署

```bash
# 启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f refinery

# 停止服务
docker-compose down
```

## 测试

```bash
pytest
pytest --cov=app tests/
pytest tests/test_rag.py -v
```

## 项目结构

```
refinery/
├── app/
│   ├── cli/           # CLI 命令与工具
│   ├── api/           # FastAPI 路由
│   ├── models/        # 数据库模型
│   ├── agents/        # LangGraph 智能体
│   ├── workflow/      # 主工作流图
│   ├── deep_search/   # ReAct 循环深度研究
│   ├── deep_graph/    # GraphRAG 构建与分析
│   ├── chat/          # 多轮对话系统
│   ├── services/      # RAG 服务
│   └── prompts/       # 集中式提示词
├── alembic/           # 数据库迁移
├── tests/             # 测试文件
├── pyproject.toml
└── docker-compose.yml
```

## 集成示例

### Cron 定时任务

```bash
# 每小时运行工作流
0 * * * * /usr/local/bin/refinery workflow run --json >> /var/log/refinery.log
```

### Python 集成

```python
import subprocess
import json

result = subprocess.run(
    ["refinery", "workflow", "run", "--json"],
    capture_output=True,
    text=True
)

data = json.loads(result.stdout)
print(f"存储了 {data['total_articles_stored']} 篇文章")
```

## 帮助

```bash
refinery --help
refinery workflow --help
refinery article --help
```

## 许可证

MIT

## 贡献指南

欢迎贡献代码！请随时提交 Pull Request。

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开 Pull Request

## 问题反馈

如遇问题，请在 https://github.com/your-org/refinery/issues 提交 issue