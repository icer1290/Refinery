[English](README.md) | [中文](README.zh-CN.md)

# 科技新闻聚合器

基于 LangGraph、FastAPI 和 PostgreSQL 构建的 AI 科技新闻聚合系统。

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

## 系统架构

### 主工作流管道

```
[Entry] → [Scout] → [Dedup] → [Scoring] → [Writing] → [Reflection] → [Storage] → [End]
```

### 深度搜索 (ReAct 循环)

按需深度研究流程：
1. 抓取文章内容
2. ReAct 循环配合网络搜索工具（DuckDuckGo 或 Tavily）
3. 生成综合报告

### GraphRAG (DeepGraph)

双阶段知识图谱系统：
- **后台 GraphBuilder**: 提取实体/关系，社区发现（Leiden 算法）
- **按需 GraphAnalyst**: 获取子图，图遍历扩展，生成分析

### 多轮对话 (Hub-and-Spoke 架构)

多智能体协作的对话式 AI：
- **Supervisor**: 中央调度器，评估意图并路由至专家智能体
- **Researcher**: ReAct 循环智能体（思考 → 工具 → 思考 → ... → 总结），深度信息收集
- **Explainer**: 提供文章解释与上下文
- **Fact Checker**: 对照知识图谱与网络来源验证观点
- **多层记忆**: 短期（会话）、中期（对话）、长期（持久化）三层记忆系统
- **自动压缩**: 接近 token 限制时自动压缩上下文

## 技术栈

- **后端**: FastAPI, LangGraph, LangChain
- **数据库**: PostgreSQL + pgvector
- **缓存**: Redis（可选，用于会话缓存）
- **AI**: OpenAI 或兼容 API（DashScope、Azure、Ollama）
- **RSS/Web**: feedparser, trafilatura, httpx, duckduckgo-search

## 安装部署

### 环境要求

- Python 3.11+
- PostgreSQL 15+ 并启用 pgvector 扩展
- OpenAI API Key（或兼容服务）

### 安装步骤

1. 克隆仓库：
```bash
git clone <repository-url>
cd ai-engine
```

2. 安装依赖（推荐使用 uv）：
```bash
uv sync
# 或使用 pip
pip install -r requirements.txt
```

3. 配置环境变量：
```bash
cp .env.example .env
# 编辑 .env 文件，填入配置
```

4. 设置数据库：
```bash
# 创建 PostgreSQL 数据库
createdb news_aggregator

# 运行迁移
alembic upgrade head
```

### 运行服务

启动服务器：
```bash
uvicorn app.main:app --reload
```

API 文档地址：http://localhost:8000/docs

### Docker 部署

```bash
docker-compose up -d
docker-compose logs -f ai-engine
docker-compose down
```

## API 接口

所有接口位于 `/api/v1/` 路径下：

### 工作流

| 接口 | 方法 | 说明 |
|------|------|------|
| `/workflow/run` | POST | 触发新闻聚合工作流 |
| `/workflow/runs` | GET | 查询工作流运行历史 |
| `/workflow/runs/{id}` | GET | 获取工作流运行详情 |
| `/workflow/articles` | GET | 获取文章列表 |
| `/workflow/articles/{id}` | GET | 获取文章详情 |
| `/workflow/feeds` | GET/POST | 查询或添加 RSS 源 |
| `/workflow/feeds/{id}` | DELETE | 删除 RSS 源 |
| `/workflow/feeds/{id}/toggle` | PATCH | 启用/禁用 RSS 源 |

### 深度搜索

| 接口 | 方法 | 说明 |
|------|------|------|
| `/deep-search/run` | POST | 对文章执行深度搜索 |

### 图谱分析

| 接口 | 方法 | 说明 |
|------|------|------|
| `/deep-graph/analyze` | POST | 生成知识图谱分析 |

### 对话

| 接口 | 方法 | 说明 |
|------|------|------|
| `/chat/conversations` | POST | 创建文章对话 |
| `/chat/conversations` | GET | 查询用户对话列表 |
| `/chat/conversations/{id}` | GET | 获取对话详情 |
| `/chat/conversations/{id}/history` | GET | 获取消息历史 |
| `/chat/chat` | POST | 发送消息并获取 AI 回复 |
| `/chat/conversations/{id}` | DELETE | 归档对话 |

### 健康检查

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/health/ready` | GET | 就绪检查 |
| `/health/live` | GET | 存活检查 |

## 配置说明

| 变量 | 必填 | 说明 |
|------|------|------|
| `DATABASE_URL` | 是 | PostgreSQL 连接 URL |
| `OPENAI_API_KEY` | 是 | OpenAI 或兼容 API Key |
| `OPENAI_CHAT_MODEL` | 是 | 聊天模型（如 gpt-4o-mini、qwen3.5-35b-a3b） |
| `OPENAI_EMBEDDING_MODEL` | 是 | 嵌入模型（如 text-embedding-3-small、text-embedding-v4） |
| `OPENAI_BASE_URL` | 否 | 自定义 API 地址（用于 DashScope、Azure、Ollama） |
| `WEB_SEARCH_PROVIDER` | 否 | 搜索引擎：「duckduckgo」或「tavily」（默认 duckduckgo） |
| `RERANK_PROVIDER` | 否 | 重排服务：「none」、「dashscope」、「cohere」、「jina」 |
| `REDIS_URL` | 否 | Redis 连接 URL（用于会话缓存） |
| `REDIS_ENABLED` | 否 | 启用 Redis 缓存（默认 true） |
| `CHAT_MAX_TOKENS` | 否 | 对话最大上下文 token（默认 254000） |
| `CHAT_CONTEXT_THRESHOLD` | 否 | 压缩阈值（默认 0.7） |
| `CHAT_SESSION_TTL` | 否 | 会话缓存 TTL 秒数（默认 1800） |
| `CHAT_MEMORY_TTL` | 否 | 记忆缓存 TTL 秒数（默认 86400） |

完整配置选项请参阅 `.env.example`。

## 测试

```bash
pytest
pytest --cov=app tests/
pytest tests/test_rag.py -v  # 运行单个测试文件
```

## 项目结构

```
ai-engine/
├── app/
│   ├── api/           # FastAPI 路由
│   ├── models/        # 数据库模型
│   ├── agents/        # LangGraph 智能体（Scout, Scorer, Writer, Reflection）
│   ├── workflow/      # 主工作流图与节点
│   ├── deep_search/   # ReAct 循环深度研究
│   ├── deep_graph/    # GraphRAG 构建器与分析器
│   ├── chat/          # 多轮对话多智能体架构
│   │   ├── agents/    # Supervisor, Researcher, Explainer, FactChecker
│   │   ├── memory/    # 多层记忆管理
│   │   └── graph.py   # Hub-and-spoke 工作流
│   ├── services/      # RAG 服务（embedding, vector_store, reranker 等）
│   ├── prompts/       # 集中式提示词模板
│   ├── core/          # 异常处理、日志
│   └── utils/         # 辅助工具、常量
├── alembic/           # 数据库迁移
├── tests/             # 测试文件
├── pyproject.toml
└── docker-compose.yml
```

## 许可证

MIT