# 科技新闻聚合 Agent 系统设计执行文档

## 1. 项目概述与技术栈

核心目标：实现从 RSS 信息源拉取、语义去重、多维价值评分、全文抓取分析到自我审查修正的全自动高质量新闻摘要工作流。

核心技术栈：
- 后端框架：FastAPI (负责 API 接口与系统服务调度)
- 工作流编排：LangGraph (利用 Send 实现动态节点并发控制)
- 数据库：PostgreSQL + pgvector (负责关系型数据与向量数据的双写持久化)

## 2. 核心工作流设计 (Workflow Architecture)

系统整体分为五个执行阶段，由 Master Agent 负责全局调度与汇总。

### Phase 1: 情报收集 (Scout Phase)

触发机制：Master Agent 启动，根据订阅的源列表，使用 LangGraph 的 Send 方法动态创建多个平行的 Scout Agent 节点。

节点行为：每个 Scout Agent 绑定 RSS 解析器工具，直接请求并解析指定的 RSS 订阅源，获取新闻的基础元数据（如 URL、初始标题、简短描述等）。

### Phase 2: 语义去重 (Deduplication Phase)

触发机制：收集到所有 Scout Agent 的初步信息后，数据流入去重节点。

节点行为：将所有收集到的信息转化为向量 (Embeddings) 放入向量数据库进行对比。

业务逻辑 (去重阈值)：仅依靠向量相似度进行判重。相似度阈值需通过测试校准到"两篇文章描述的是同一个核心事件"的语义级别，超过该阈值的重复资讯将被丢弃，仅保留最完整的一条进入下一环节。

### Phase 3: 价值评估 (Scoring Phase)

触发机制：针对去重后的资讯列表，再次动态拉起多个并行的 Scorer Agent。

节点行为：读取向量数据库中保留的资讯内容，对每一篇内容进行价值打分。

业务逻辑 (评分维度)：Prompt 需强制 LLM 输出 JSON 格式的评分（1-10分），并综合以下三个维度：
- 行业影响力：该事件对相关行业格局的潜在影响程度。
- 关键节点性：是否属于技术突破、产品发布、重大融资等里程碑事件。
- 引人关注度：话题的话题性、公众兴趣及传播潜力。

(注：需在开发时设定一个总分阈值，例如总分低于 6 分的文章将被直接阻断，不进入消耗 Token 更多的 Writer 阶段)。

### Phase 4: 深度撰写与审查 (Writing & Reflection Phase)

触发机制：高分文章列表返回给 Master Agent 后，为每篇文章动态创建 Writer Agent。

节点行为 (Writer)：调用 extract web 工具，传入文章 URL 抓取网页完整正文。基于全文生成包含中文标题和清晰中文摘要的短报。

节点行为 (Reflection)：生成的内容必须经过 Reflection（反思/自省）节点的校验。如果未通过校验，则打回 Writer 节点重新生成。

业务逻辑 (校验标准)：Reflection 节点的 Prompt 将作为严格的"裁判"，校验以下核心规则：
- 格式合规：必须包含中文标题和中文摘要。
- 实体保留：翻译过程中，必须避开人名、公司名、产品名（需保留原语言）。
- 客观中立：必须消除主观偏见，保持新闻报道的客观陈述口吻。

(注：建议在 LangGraph 中设定 max_retries = 3，若重试 3 次仍未通过则抛出异常或人工标记)。

### Phase 5: 最终存储 (Storage Phase)

触发机制：所有通过 Reflection 审查的文章由最终的 Master Agent 统一收集。

节点行为：执行数据库"双写"操作。将结构化的短报数据、标题、URL 同时写入 PostgreSQL（用于业务展示）和 pgvector（用于后续的历史检索和知识库功能）。

# 参考信息
## RSS feeds
```
SS_FEEDS = [
    # --- 核心科技媒体 (全领域覆盖) ---
    {"name": "Techmeme", "url": "https://www.techmeme.com/feed.xml"},
    {"name": "The Verge", "url": "https://www.theverge.com/rss/index.xml"},
    {"name": "TechCrunch", "url": "https://techcrunch.com/feed/"},
    {"name": "Wired (Science)", "url": "https://www.wired.com/feed/category/science/latest/rss"},
    
    # --- 顶级风投与趋势 (宏观视角/范式转移) ---
    {"name": "a16z", "url": "https://www.a16z.news/feed"},
    {"name": "Y Combinator", "url": "https://blog.ycombinator.com/feed/"},
    {"name": "Sequoia Capital", "url": "https://www.sequoiacap.com/feed/"},
    
    # --- AI 与巨头官报 (第一手重磅消息) ---
    {"name": "OpenAI Blog", "url": "https://openai.com/news/rss.xml"},
    {"name": "Google Blog", "url": "https://blog.google/rss/"},
    {"name": "Microsoft Blog", "url": "https://blogs.microsoft.com/feed/"},
    {"name": "Meta Newsroom", "url": "https://about.fb.com/news/feed/"},
    {"name": "NVIDIA Blog", "url": "https://blogs.nvidia.com/feed/"},
    
    # --- 硬核工程与架构 (技术创新度) ---
    {"name": "Netflix Tech Blog", "url": "https://netflixtechblog.com/feed"},
    {"name": "GitHub Engineering", "url": "https://github.blog/category/engineering/feed/"},
    {"name": "Cloudflare Blog", "url": "https://blog.cloudflare.com/rss/"},
    {"name": "InfoQ", "url": "https://www.infoq.com/feed"},
    {"name": "The New Stack", "url": "https://thenewstack.io/blog/feed/"},
    # {"name": "Dev.to", "url": "https://dev.to/feed"},
    
    # --- 深度专栏与分析 (主编视角洞察) ---
    {"name": "Stratechery (Ben Thompson)", "url": "https://stratechery.com/feed/"},
    {"name": "The Pragmatic Engineer", "url": "https://blog.pragmaticengineer.com/rss/"}
]
```
