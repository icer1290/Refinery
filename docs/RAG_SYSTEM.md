# RAG 系统完整文档

本文档详细介绍项目中 RAG（Retrieval-Augmented Generation）系统的所有功能和实现方式。

---

## 目录

1. [系统概述](#1-系统概述)
2. [数据库设计](#2-数据库设计)
3. [核心服务模块](#3-核心服务模块)
4. [Workflow 集成](#4-workflow-集成)
5. [DeepSearch 集成](#5-deepsearch-集成)
6. [配置参考](#6-配置参考)
7. [使用示例](#7-使用示例)

---

## 1. 系统概述

### 1.1 架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              索引 Pipeline                                   │
└─────────────────────────────────────────────────────────────────────────────┘

    Article Content ──▶ ChunkingService ──▶ EmbeddingService ──▶ VectorStore
         │                    │                      │                   │
         │                    ▼                      ▼                   ▼
         │              TextChunk[]            embeddings[]      store_chunk_embeddings()
         │
         └──────────────────────────────────────────────────────────────────┐
                                                                              │
┌─────────────────────────────────────────────────────────────────────────────┐
│                              检索 Pipeline                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                                                              │
                          ┌──────────────────┐                               │
                          │  User Query      │                               │
                          └────────┬─────────┘                               │
                                   │                                         │
                                   ▼                                         │
                    ┌──────────────────────────────┐                         │
                    │   QueryTransformService      │                         │
                    │   - HyDE                     │                         │
                    │   - Multi-query expansion    │                         │
                    │   - Keyword extraction       │                         │
                    └──────────────┬───────────────┘                         │
                                   │                                         │
                                   ▼                                         │
                    ┌──────────────────────────────┐                         │
                    │     EmbeddingService         │                         │
                    │     - embed_text()           │                         │
                    │     - embed_batch()          │                         │
                    └──────────────┬───────────────┘                         │
                                   │                                         │
                                   ▼                                         │
                    ┌──────────────────────────────┐                         │
                    │       VectorStore            │                         │
                    │   - hybrid_search()          │◀────────────────────────┘
                    │   - vector_search_chunks()   │
                    │   (RRF fusion)               │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │     RerankerService          │
                    │   - rerank()                 │
                    │   (Cross-encoder)            │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │    CompressionService        │
                    │   - compress_chunks()        │
                    │   - extract_key_sentences()  │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                          ┌──────────────────┐
                          │  LLM Generator   │
                          └──────────────────┘
```

### 1.2 核心组件

| 组件 | 文件路径 | 职责 |
|------|----------|------|
| ChunkingService | `app/services/chunking.py` | 文本切分，保留语义边界 |
| EmbeddingService | `app/services/embedding.py` | 文本向量化 |
| VectorStore | `app/services/vector_store.py` | 向量存储与混合检索 |
| RerankerService | `app/services/reranker.py` | 检索结果重排序 |
| QueryTransformService | `app/services/query_transform.py` | 查询变换与扩展 |
| CompressionService | `app/services/compression.py` | 上下文压缩 |

---

## 2. 数据库设计

### 2.1 数据模型

#### NewsArticle 模型

**文件**: `app/models/orm_models.py`

```python
class NewsArticle(Base):
    """新闻文章模型。"""
    __tablename__ = "news_articles"

    # 主键
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # 来源信息
    source_name: Mapped[str] = mapped_column(String(255), nullable=False)
    source_url: Mapped[str] = mapped_column(String(2048), unique=True, nullable=False)

    # 原始内容
    original_title: Mapped[str] = mapped_column(Text, nullable=False)
    original_description: Mapped[str | None] = mapped_column(Text)

    # 翻译内容
    chinese_title: Mapped[str | None] = mapped_column(Text)
    chinese_summary: Mapped[str | None] = mapped_column(Text)
    full_content: Mapped[str | None] = mapped_column(Text)

    # 评分
    total_score: Mapped[float | None] = mapped_column(Float)
    industry_impact_score: Mapped[float | None] = mapped_column(Float)
    milestone_score: Mapped[float | None] = mapped_column(Float)
    attention_score: Mapped[float | None] = mapped_column(Float)

    # 时间戳
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    processed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Reflection 验证
    reflection_retries: Mapped[int] = mapped_column(Integer, default=0)
    reflection_passed: Mapped[bool] = mapped_column(Boolean, default=False)
    reflection_feedback: Mapped[str | None] = mapped_column(Text)
    is_published: Mapped[bool] = mapped_column(Boolean, default=False)

    # DeepSearch 报告
    deepsearch_report: Mapped[str | None] = mapped_column(Text)
    deepsearch_performed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # 关系：一对多（一篇文章有多个 embedding）
    embeddings: Mapped[list["ArticleEmbedding"]] = relationship(
        "ArticleEmbedding", back_populates="article", cascade="all, delete-orphan"
    )
```

#### ArticleEmbedding 模型

```python
class ArticleEmbedding(Base):
    """文章向量模型，支持 pgvector。

    支持两种类型的嵌入：
    - summary: 标题 + 描述的嵌入（用于去重）
    - chunk: 正文片段的嵌入（用于 RAG 检索）
    """
    __tablename__ = "article_embeddings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # 外键关联
    article_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("news_articles.id", ondelete="CASCADE"),
        nullable=False,
    )

    # 向量数据（1024 维）
    embedding: Mapped[list[float]] = mapped_column(Vector(1024), nullable=False)

    # 内容哈希（用于去重）
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    # 创建时间
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # ===== Chunk 特有字段（RAG 检索）=====

    # Chunk 编号（summary=-1, chunk=0,1,2...）
    chunk_number: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Chunk 文本内容
    chunk_text: Mapped[str | None] = mapped_column(Text)

    # 字符偏移量（用于定位原文位置）
    chunk_start: Mapped[int | None] = mapped_column(Integer)
    chunk_end: Mapped[int | None] = mapped_column(Integer)

    # 嵌入类型：'summary' | 'chunk'
    embedding_type: Mapped[str] = mapped_column(
        String(20), default="summary", nullable=False
    )

    # 关系
    article: Mapped["NewsArticle"] = relationship(
        "NewsArticle", back_populates="embeddings"
    )

    # 索引
    __table_args__ = (
        # 复合唯一索引：每篇文章的每个 chunk 只能有一条记录
        Index("ix_article_embeddings_article_chunk", article_id, chunk_number, unique=True),
        Index("ix_article_embeddings_content_hash", content_hash),
    )
```

### 2.2 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `embedding` | Vector(1024) | 1024 维向量，使用 DashScope text-embedding-v4 |
| `content_hash` | String(64) | SHA256 哈希，用于检测内容变化 |
| `chunk_number` | Integer | Chunk 编号，summary=-1，chunk 从 0 开始 |
| `chunk_text` | Text | Chunk 的文本内容 |
| `chunk_start` | Integer | 在原文中的起始字符位置 |
| `chunk_end` | Integer | 在原文中的结束字符位置 |
| `embedding_type` | String(20) | "summary" 或 "chunk" |

### 2.3 数据库迁移

**文件**: `alembic/versions/002_add_chunk_embeddings.py`

迁移脚本实现了以下功能：

```python
def upgrade() -> None:
    # 1. 移除 article_id 的 unique 约束（允许一篇文章多个 embedding）
    op.drop_constraint('article_embeddings_article_id_key', 'article_embeddings', type_='unique')

    # 2. 添加 Chunk 相关字段
    op.add_column('article_embeddings', sa.Column('chunk_number', sa.Integer, default=0))
    op.add_column('article_embeddings', sa.Column('chunk_text', sa.Text))
    op.add_column('article_embeddings', sa.Column('chunk_start', sa.Integer))
    op.add_column('article_embeddings', sa.Column('chunk_end', sa.Integer))
    op.add_column('article_embeddings', sa.Column('embedding_type', sa.String(20), default='summary'))

    # 3. 创建复合唯一索引
    op.create_index(
        'ix_article_embeddings_article_chunk',
        'article_embeddings',
        ['article_id', 'chunk_number'],
        unique=True
    )

    # 4. 添加 TSVECTOR 列（全文检索）
    op.add_column('article_embeddings', sa.Column('chunk_tsv', postgresql.TSVECTOR))

    # 5. 创建 GIN 索引（加速全文检索）
    op.create_index(
        'ix_article_embeddings_chunk_tsv',
        'article_embeddings',
        ['chunk_tsv'],
        postgresql_using='gin'
    )

    # 6. 创建触发器函数（自动更新 tsvector）
    op.execute("""
        CREATE OR REPLACE FUNCTION update_chunk_tsv()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.chunk_tsv :=
                setweight(to_tsvector('english', COALESCE(NEW.chunk_text, '')), 'A') ||
                setweight(to_tsvector('english', COALESCE(NEW.chunk_text, '')), 'B');
            RETURN NEW;
        END
        $$ LANGUAGE plpgsql
    """)

    # 7. 创建触发器
    op.execute("""
        CREATE TRIGGER chunk_tsv_update
            BEFORE INSERT OR UPDATE ON article_embeddings
            FOR EACH ROW
            WHEN (pg_trigger_depth() = 0)
            EXECUTE FUNCTION update_chunk_tsv()
    """)

    # 8. 优化向量索引（IVFFlat）
    op.execute("""
        CREATE INDEX ix_article_embeddings_vector
        ON article_embeddings
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    """)
```

### 2.4 索引设计

| 索引名 | 类型 | 用途 |
|--------|------|------|
| `ix_article_embeddings_article_chunk` | B-tree | 复合唯一索引，确保每篇文章每个 chunk 唯一 |
| `ix_article_embeddings_content_hash` | B-tree | 内容哈希索引，用于快速查重 |
| `ix_article_embeddings_chunk_tsv` | GIN | 全文检索索引 |
| `ix_article_embeddings_vector` | IVFFlat | 向量相似度索引，加速最近邻搜索 |

---

## 3. 核心服务模块

### 3.1 ChunkingService（文本切分）

**文件**: `app/services/chunking.py`

#### 类签名

```python
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter

@dataclass
class TextChunk:
    """文本片段数据类。"""
    text: str           # Chunk 文本内容
    chunk_number: int   # Chunk 编号
    start_char: int     # 起始字符位置
    end_char: int       # 结束字符位置


class ChunkingService:
    """文本切分服务，用于将长文本分割成适合嵌入的小片段。"""

    def __init__(
        self,
        chunk_size: int | None = None,      # 默认 2000 字符
        chunk_overlap: int | None = None,   # 默认 400 字符
    ): ...

    def chunk_text(self, text: str) -> list[TextChunk]:
        """将文本切分为多个片段。"""
        ...

    def chunk_text_with_summary_first(
        self,
        text: str,
        summary: str,
        max_summary_chunk: int = 500,
    ) -> list[TextChunk]:
        """将摘要前置到第一个 chunk，提高长文档检索效果。"""
        ...
```

#### 算法原理：递归字符切分

使用 LangChain 的 `RecursiveCharacterTextSplitter`，按分隔符优先级依次尝试切分：

```python
separators = [
    "\n\n\n",  # 1. 优先：大章节
    "\n\n",    # 2. 段落
    "\n",      # 3. 行
    ". ",      # 4. 句子
    " ",       # 5. 单词
    "",        # 6. 最后手段：字符级别
]
```

**工作原理**：
1. 尝试用第一个分隔符切分
2. 如果切分后仍有 chunk 超过 `chunk_size`，用下一个分隔符继续切分
3. 重复直到所有 chunk 都满足大小要求

#### Summary-First 策略

将文章摘要前置到第一个 chunk，提高检索效果：

```python
def chunk_text_with_summary_first(self, text: str, summary: str, max_summary_chunk: int = 500):
    chunks = self.chunk_text(text)

    # 截断摘要
    truncated_summary = summary[:max_summary_chunk] if len(summary) > max_summary_chunk else summary
    summary_prefix = f"[摘要] {truncated_summary}\n\n---\n\n"

    # 如果第一个 chunk 有足够空间，合并摘要
    if len(summary_prefix) + len(chunks[0].text) <= self.chunk_size:
        chunks[0] = TextChunk(
            text=summary_prefix + chunks[0].text,
            chunk_number=0,
            start_char=chunks[0].start_char,
            end_char=chunks[0].end_char,
        )
    else:
        # 否则插入为独立的第一个 chunk
        chunks.insert(0, TextChunk(
            text=summary_prefix.rstrip(),
            chunk_number=0,
            start_char=0,
            end_char=len(summary_prefix),
        ))
```

**优势**：长文档的摘要通常包含关键信息，将其放在第一个 chunk 可以提高检索召回率。

#### 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rag_chunk_size` | 2000 | 每个 chunk 的最大字符数 |
| `rag_chunk_overlap` | 400 | 相邻 chunk 的重叠字符数 |

---

### 3.2 EmbeddingService（向量嵌入）

**文件**: `app/services/embedding.py`

#### 类签名

```python
class EmbeddingService:
    """文本向量化服务，支持 OpenAI 和 DashScope API。"""

    def __init__(
        self,
        model: str | None = None,        # 默认 "text-embedding-v4"
        api_key: str | None = None,      # 从环境变量读取
        base_url: str | None = None,     # 支持 DashScope
    ): ...

    async def embed_text(self, text: str) -> list[float]:
        """生成单个文本的向量嵌入。"""
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量生成多个文本的向量嵌入。"""
        ...

    @staticmethod
    async def compute_similarity(embedding1: list[float], embedding2: list[float]) -> float:
        """计算两个向量的余弦相似度。"""
        ...
```

#### 多提供商支持

自动检测 DashScope 并使用对应的 API 格式：

```python
def __init__(self, model, api_key, base_url):
    self.model = model or settings.openai_embedding_model
    self.api_key = api_key or settings.openai_api_key
    self.base_url = base_url or settings.openai_base_url

    # 检测是否使用 DashScope
    self._is_dashscope = self.base_url and "dashscope" in self.base_url.lower()


async def embed_text(self, text: str) -> list[float]:
    truncated_text = self._truncate_text(text)

    if self._is_dashscope:
        return await self._embed_with_dashscope(truncated_text)
    else:
        return await self._embed_with_openai(truncated_text)
```

#### DashScope API 调用

```python
async def _embed_with_dashscope(self, text: str) -> list[float]:
    """DashScope 嵌入 API 格式。"""
    url = f"{self.base_url}/embeddings"

    headers = {
        "Authorization": f"Bearer {self.api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": self.model,
        "input": {"texts": [text]},
        "parameters": {"text_type": "document"},
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        data = response.json()
        return data["output"]["embeddings"][0]["embedding"]
```

#### 文本截断

防止超出 API token 限制：

```python
def _truncate_text(self, text: str, max_tokens: int = 8000) -> str:
    """截断文本以适应 token 限制。"""
    # 粗略估算：1 token ≈ 4 字符
    max_chars = max_tokens * 4
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text
```

#### 重试机制

使用 tenacity 实现自动重试：

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
async def embed_text(self, text: str) -> list[float]:
    ...
```

#### 余弦相似度计算

```python
@staticmethod
async def compute_similarity(embedding1: list[float], embedding2: list[float]) -> float:
    """计算余弦相似度。"""
    dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
    mag1 = sum(a * a for a in embedding1) ** 0.5
    mag2 = sum(b * b for b in embedding2) ** 0.5
    return dot_product / (mag1 * mag2)
```

---

### 3.3 VectorStore（向量存储与检索）

**文件**: `app/services/vector_store.py`

#### 类签名

```python
@dataclass
class SearchResult:
    """混合检索结果。"""
    article_id: uuid.UUID
    chunk_text: str
    similarity: float
    article_title: str
    article_summary: str
    source_name: str
    source_url: str
    chunk_number: int
    embedding_type: str


class VectorStore:
    """PostgreSQL pgvector 操作类。"""

    def __init__(self, similarity_threshold: float = 0.85): ...

    async def store_embedding(
        self,
        session: AsyncSession,
        article_id: uuid.UUID,
        embedding: list[float],
        content_hash: str,
    ) -> ArticleEmbedding:
        """存储 summary embedding。"""
        ...

    async def store_chunk_embeddings(
        self,
        session: AsyncSession,
        article_id: uuid.UUID,
        chunks: list[tuple[str, int, int, list[float]]],  # (text, start, end, embedding)
        content_hash: str,
    ) -> list[ArticleEmbedding]:
        """批量存储 chunk embeddings。"""
        ...

    async def hybrid_search(
        self,
        session: AsyncSession,
        query: str,
        embedding: list[float],
        limit: int = 10,
        vector_weight: float = 0.6,
        fts_weight: float = 0.4,
    ) -> list[SearchResult]:
        """混合检索：向量相似度 + 全文检索。"""
        ...

    async def vector_search_chunks(
        self,
        session: AsyncSession,
        embedding: list[float],
        limit: int = 10,
        similarity_threshold: float = 0.5,
    ) -> list[SearchResult]:
        """纯向量检索。"""
        ...

    async def find_similar(
        self,
        session: AsyncSession,
        embedding: list[float],
        limit: int = 10,
        exclude_ids: list[uuid.UUID] | None = None,
        similarity_threshold: float = 0.85,
    ) -> list[tuple[NewsArticle, float]]:
        """查找相似文章（基于 summary embedding）。"""
        ...
```

#### 混合检索算法：RRF（Reciprocal Rank Fusion）

混合检索结合向量相似度和全文检索，使用 RRF 融合排名：

```sql
-- 向量检索 CTE
WITH vector_results AS (
    SELECT
        ae.article_id,
        ae.chunk_text,
        ae.chunk_number,
        ae.embedding_type,
        -- 计算余弦相似度（1 - 余弦距离）
        1 - (ae.embedding <=> CAST(:embedding AS vector)) as vector_score,
        -- 按相似度排名
        ROW_NUMBER() OVER (ORDER BY ae.embedding <=> CAST(:embedding AS vector)) as vector_rank
    FROM article_embeddings ae
    WHERE ae.embedding_type = 'chunk'
    ORDER BY ae.embedding <=> CAST(:embedding AS vector)
    LIMIT :limit * 2
),

-- 全文检索 CTE
fts_results AS (
    SELECT
        ae.article_id,
        ae.chunk_text,
        ae.chunk_number,
        ae.embedding_type,
        -- 计算全文检索得分
        ts_rank(ae.chunk_tsv, plainto_tsquery('english', :query)) as fts_score,
        -- 按得分排名
        ROW_NUMBER() OVER (ORDER BY ts_rank(ae.chunk_tsv, plainto_tsquery('english', :query)) DESC) as fts_rank
    FROM article_embeddings ae
    WHERE ae.embedding_type = 'chunk'
      AND ae.chunk_tsv @@ plainto_tsquery('english', :query)
    ORDER BY fts_score DESC
    LIMIT :limit * 2
),

-- 合并结果
combined AS (
    SELECT
        COALESCE(v.article_id, f.article_id) as article_id,
        COALESCE(v.chunk_text, f.chunk_text) as chunk_text,
        COALESCE(v.chunk_number, f.chunk_number) as chunk_number,
        COALESCE(v.embedding_type, f.embedding_type) as embedding_type,
        COALESCE(v.vector_rank, 1000) as vector_rank,
        COALESCE(f.fts_rank, 1000) as fts_rank
    FROM vector_results v
    FULL OUTER JOIN fts_results f
        ON v.article_id = f.article_id AND v.chunk_number = f.chunk_number
)

-- RRF 融合公式
SELECT
    c.*,
    na.original_title as article_title,
    na.chinese_summary as article_summary,
    na.source_name,
    na.source_url
FROM combined c
JOIN news_articles na ON c.article_id = na.id
ORDER BY
    -- RRF 公式：权重 / (排名 + k)，k 通常取 60
    :vector_weight / (c.vector_rank + 60) +
    :fts_weight / (c.fts_rank + 60) DESC
LIMIT :limit
```

#### RRF 公式说明

```
RRF_score(d) = Σ (w_i / (rank_i(d) + k))
```

- `w_i`: 各检索方法的权重
- `rank_i(d)`: 文档 d 在方法 i 中的排名
- `k`: 平滑常数，通常为 60

**优势**：
- 无需归一化分数
- 对排名波动不敏感
- 能有效结合不同检索方法的结果

#### 存储操作

```python
async def store_chunk_embeddings(
    self,
    session: AsyncSession,
    article_id: uuid.UUID,
    chunks: list[tuple[str, int, int, list[float]]],
    content_hash: str,
) -> list[ArticleEmbedding]:
    """存储 chunk embeddings。"""

    # 1. 删除旧的 chunk embeddings
    await session.execute(
        text("DELETE FROM article_embeddings WHERE article_id = :article_id AND embedding_type = 'chunk'"),
        {"article_id": str(article_id)},
    )

    # 2. 插入新的 chunk embeddings
    embeddings = []
    for i, (chunk_text, start_char, end_char, embedding) in enumerate(chunks):
        # 生成唯一的 content_hash（截取前 60 字符避免超长）
        chunk_hash = f"{content_hash[:60]}_c{i}"

        db_embedding = ArticleEmbedding(
            article_id=article_id,
            embedding=embedding,
            content_hash=chunk_hash,
            embedding_type="chunk",
            chunk_number=i,
            chunk_text=chunk_text,
            chunk_start=start_char,
            chunk_end=end_char,
        )
        session.add(db_embedding)
        embeddings.append(db_embedding)

    return embeddings
```

---

### 3.4 RerankerService（重排序）

**文件**: `app/services/reranker.py`

#### 类签名

```python
class RerankerService:
    """使用 Cross-Encoder 模型对检索结果重排序。"""

    def __init__(
        self,
        model: str | None = None,        # 默认 "gte-rerank"
        api_key: str | None = None,
        base_url: str | None = None,
    ): ...

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[tuple[SearchResult, float]]:
        """重排序检索结果。

        Returns:
            List of (SearchResult, relevance_score) tuples
        """
        ...
```

#### Cross-Encoder 原理

与 Bi-Encoder（如向量检索）不同，Cross-Encoder 将 query 和 document 一起输入模型，计算精确的相关性得分：

```
Bi-Encoder:  Query → [Embedding]  Document → [Embedding]  → Cosine Similarity
Cross-Encoder: [Query + Document] → [Model] → Relevance Score
```

**优势**：Cross-Encoder 更准确，但速度较慢，适合对少量候选结果精排。

#### DashScope Rerank API

```python
async def _call_dashscope_rerank(
    self,
    query: str,
    documents: list[str],
) -> list[float]:
    """调用 DashScope Rerank API。

    注意：DashScope Rerank 使用原生 API 端点，不是 OpenAI 兼容模式。
    """
    url = "https://dashscope.aliyuncs.com/api/v1/services/rerank"

    headers = {
        "Authorization": f"Bearer {self.api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": self.model,  # "gte-rerank"
        "input": {
            "query": query,
            "documents": documents,
        },
        "parameters": {
            "return_documents": False,  # 不返回文档内容，只返回得分
        },
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        data = response.json()

        # 响应格式：{"output": {"results": [{"index": 0, "relevance_score": 0.9}, ...]}}
        results = data["output"]["results"]
        results.sort(key=lambda x: x["index"])  # 按原顺序排列
        return [r["relevance_score"] for r in results]
```

#### 降级处理

```python
async def rerank(self, query: str, results: list[SearchResult], top_k: int | None = None):
    try:
        scores = await self._call_rerank_api(query, documents)
        scored_results = list(zip(results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results[:top_k]
    except Exception as e:
        logger.error("Reranking failed, falling back to original order", error=str(e))
        # 降级：返回原始顺序
        return [(r, r.similarity) for r in results[:top_k]]
```

---

### 3.5 QueryTransformService（查询变换）

**文件**: `app/services/query_transform.py`

#### 类签名

```python
class QueryTransformService:
    """查询变换服务，提高检索召回率。"""

    async def generate_hypothetical_document(
        self,
        query: str,
        doc_length: int = 500,
    ) -> str:
        """HyDE: 生成假设性文档。"""
        ...

    async def expand_query(self, query: str, n: int = 3) -> list[str]:
        """多查询扩展。"""
        ...

    async def extract_keywords(self, query: str, n: int = 5) -> list[str]:
        """关键词提取。"""
        ...
```

#### HyDE（Hypothetical Document Embedding）

生成一个假设性文档，该文档能够完美回答用户的问题，然后对该文档进行嵌入和检索：

```python
async def generate_hypothetical_document(self, query: str, doc_length: int = 500) -> str:
    prompt = f"""请生成一篇假设性的新闻文章，这篇文章能够完美回答以下问题。
文章应该包含详细的技术细节和背景信息。
目标长度约{doc_length}字。

问题: {query}

请直接输出文章内容，不要添加任何解释或标注:"""

    response = await self._call_llm(prompt)
    return response
```

**原理**：
- 用户查询可能很短或模糊
- 生成假设性文档可以将查询扩展为更丰富的上下文
- 对假设文档嵌入检索可以找到语义更相关的结果

#### 多查询扩展

从不同角度生成多个相关查询：

```python
async def expand_query(self, query: str, n: int = 3) -> list[str]:
    prompt = f"""你是一个搜索助手。请将以下查询扩展为{n}个相关但不同的搜索查询。
这些查询应该从不同角度探索原始问题的不同方面。

原始查询: {query}

请以JSON数组格式输出，例如:
["查询1", "查询2", "查询3"]
"""

    response = await self._call_llm(prompt)
    return json.loads(response)
```

---

### 3.6 CompressionService（上下文压缩）

**文件**: `app/services/compression.py`

#### 类签名

```python
class CompressionService:
    """上下文压缩服务，减少检索结果的噪声。"""

    async def compress_chunks(
        self,
        query: str,
        results: list[SearchResult],
        max_length: int = 2000,
    ) -> str:
        """压缩多个 chunk 为精简上下文。"""
        ...

    async def extract_key_sentences(
        self,
        query: str,
        results: list[SearchResult],
        max_sentences: int = 10,
    ) -> list[str]:
        """提取关键句子。"""
        ...

    async def summarize_for_context(
        self,
        query: str,
        results: list[SearchResult],
    ) -> str:
        """为上下文总结。"""
        ...
```

#### 上下文压缩实现

```python
async def compress_chunks(
    self,
    query: str,
    results: list[SearchResult],
    max_length: int = 2000,
) -> str:
    """压缩多个 chunk。"""

    # 合并所有 chunk 文本
    chunks_text = "\n\n---\n\n".join([
        f"【来源: {r.source_name}】\n{r.chunk_text}"
        for r in results
    ])

    prompt = f"""你是一个信息提取助手。请从以下多个文档片段中提取与问题直接相关的信息。

要求:
1. 只提取与问题直接相关的内容
2. 保留关键技术细节、数字、名称等具体信息
3. 去除无关的背景信息和冗余内容
4. 保持信息的准确性，不要添加文中没有的内容
5. 如果多个来源提到相同信息，可以合并
6. 在信息后标注来源，格式: (来源名称)

问题: {query}

文档片段:
{chunks_text}

请输出提取的信息（不超过{max_length}字）:"""

    return await self._call_llm(prompt)
```

**用途**：
- 减少 LLM 输入 token 数
- 去除无关信息
- 保留来源标注

---

## 4. Workflow 集成

### 4.1 Pipeline 流程

```
Entry → Scout (RSS fetch) → Dedup → Scoring (AI) → Writing (extract + translate) → Reflection (validate) → Storage (embeddings) → End
```

### 4.2 storage_node 实现

**文件**: `app/workflow/nodes.py`

```python
async def storage_node(
    state: WorkflowState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """存储文章并创建嵌入向量。"""

    session = runtime.context.session
    articles = state["final_articles"]

    stored_ids = []
    embedding_service = get_embedding_service()
    chunking_service = get_chunking_service()

    for article in articles:
        try:
            # 使用嵌套事务（savepoint）确保单篇文章失败不影响其他文章
            async with session.begin_nested():
                # 1. 创建文章记录
                db_article = NewsArticle(
                    source_name=article["source_name"],
                    source_url=article["source_url"],
                    original_title=article["original_title"],
                    original_description=article.get("original_description"),
                    chinese_title=article.get("chinese_title"),
                    chinese_summary=article.get("chinese_summary"),
                    full_content=article.get("full_content"),
                    # ... 其他字段
                )
                session.add(db_article)
                await session.flush()

                # 2. 创建 Summary Embedding
                summary_content = f"{article['original_title']} {article.get('original_description', '')}"
                content_hash = hashlib.sha256(summary_content.encode()).hexdigest()
                summary_embedding = await embedding_service.embed_text(summary_content)
                await vector_store.store_embedding(
                    session=session,
                    article_id=db_article.id,
                    embedding=summary_embedding,
                    content_hash=content_hash,
                )

                # 3. 创建 Chunk Embeddings
                full_content = article.get("full_content")
                if full_content and len(full_content.strip()) > 100:
                    chinese_summary = article.get("chinese_summary", "")

                    # 切分文本
                    chunks = chunking_service.chunk_text_with_summary_first(
                        text=full_content,
                        summary=chinese_summary,
                    )

                    if chunks:
                        # 批量生成嵌入
                        chunk_texts = [c.text for c in chunks]
                        chunk_embeddings = await embedding_service.embed_batch(chunk_texts)

                        # 准备存储数据
                        chunk_data = [
                            (c.text, c.start_char, c.end_char, emb)
                            for c, emb in zip(chunks, chunk_embeddings)
                        ]

                        # 存储 chunk embeddings
                        article_content_hash = hashlib.sha256(full_content.encode()).hexdigest()
                        await vector_store.store_chunk_embeddings(
                            session=session,
                            article_id=db_article.id,
                            chunks=chunk_data,
                            content_hash=article_content_hash,
                        )

                stored_ids.append(str(db_article.id))

        except Exception as e:
            logger.error("Failed to store article", url=article.get("source_url"), error=str(e))
            # 嵌套事务自动回滚，继续处理下一篇文章
            continue

    return {
        "stored_article_ids": stored_ids,
        "current_phase": "storage_complete",
        "total_articles_stored": len(stored_ids),
    }
```

---

## 5. DeepSearch 集成

### 5.1 VectorSearchTool

**文件**: `app/deep_search/tools.py`

```python
class VectorSearchTool(BaseTool):
    """向量检索工具，供 DeepSearch ReAct 循环调用。"""

    name = "vector_search"

    async def execute(
        self,
        session: AsyncSession,
        query: str,
        limit: int = 5,
        use_rerank: bool = True,
        use_compression: bool = False,
        use_hybrid: bool = True,
    ) -> str:
        """执行向量检索。"""

        # 1. 生成查询嵌入
        embedding_service = get_embedding_service()
        embedding = await embedding_service.embed_text(query)

        # 2. 执行检索
        if use_hybrid:
            results = await vector_store.hybrid_search(
                session=session,
                query=query,
                embedding=embedding,
                limit=settings.rag_rerank_top_k,  # 先获取更多候选
            )
        else:
            results = await vector_store.vector_search_chunks(
                session=session,
                embedding=embedding,
                limit=settings.rag_rerank_top_k,
            )

        # 3. 降级：如果没有 chunk 结果，尝试 summary 检索
        if not results:
            articles = await vector_store.find_similar(
                session=session,
                embedding=embedding,
                limit=limit,
                similarity_threshold=0.6,
            )
            # 转换为 SearchResult 格式...

        # 4. 重排序
        if use_rerank and len(results) > limit:
            reranker = get_reranker_service()
            ranked_results = await reranker.rerank(
                query=query,
                results=results,
                top_k=limit,
            )
            results = [r for r, _ in ranked_results]

        # 5. 上下文压缩（可选）
        if use_compression and results:
            compression_service = get_compression_service()
            compressed = await compression_service.compress_chunks(
                query=query,
                results=results[:limit],
            )
            return self._format_compressed_results(query, results[:limit], compressed)

        return self._format_results(results[:limit])
```

### 5.2 ReAct 循环

**文件**: `app/deep_search/prompts.py`

```python
REACT_SYSTEM_PROMPT = """你是一个深度新闻分析助手，使用 ReAct 方法来收集新闻的背景信息。

## 可用工具

1. **vector_search** - 在本地数据库中搜索相关文章（优先使用）
   - 输入: {"query": "搜索查询", "limit": 5}
   - 用途: 查找历史相关报道、背景文章、技术细节

2. **web_search** - 在网络上搜索相关信息
   - 输入: {"query": "搜索查询"}
   - 用途: 获取最新外部信息、官方声明、实时新闻

## 工作流程

1. **优先本地**: 先使用 vector_search 查找本地相关文章
2. **补充外部**: 如果本地信息不足，再使用 web_search
3. **思考 (Thought)**: 分析当前信息，决定下一步行动
4. **行动 (Action)**: 选择工具并执行
5. **观察 (Observation)**: 分析工具返回的结果
6. 重复直到收集足够信息

## 输出格式

每次回复必须是一个JSON对象:
- thought: 你的思考过程
- action: "vector_search" 或 "web_search" 或 "conclude"
- action_input: 工具输入
"""
```

### 5.3 工作流编排

**文件**: `app/deep_search/graph.py`

```python
async def run_deep_search(
    session: AsyncSession,
    article_id: str,
    max_iterations: int = 5,
) -> DeepSearchState:
    """执行 DeepSearch 工作流。"""

    # 1. 创建初始状态
    state = create_initial_deep_search_state(
        article_id=article_id,
        max_iterations=max_iterations,
    )

    # 2. 获取文章信息
    state.update(await fetch_article_node(state, session))

    # 3. ReAct 循环
    while state["should_continue"] and state["current_iteration"] < state["max_iterations"]:
        # 推理步骤
        state.update(await reasoning_node(state))

        if not state["should_continue"]:
            break

        # 工具执行
        if state.get("_pending_action"):
            state.update(await tools_node(state, session))

    # 4. 生成报告
    state.update(await conclude_node(state))

    # 5. 保存结果
    if state.get("is_complete") and state.get("final_report"):
        article.deepsearch_report = state["final_report"]
        article.deepsearch_performed_at = datetime.now(timezone.utc)
        await session.commit()

    return state
```

---

## 6. 配置参考

**文件**: `app/config.py`

```python
class Settings(BaseSettings):
    # ===== LLM API 配置 =====
    openai_api_key: str = ""
    openai_base_url: Optional[str] = None  # DashScope: https://dashscope.aliyuncs.com/compatible-mode/v1
    openai_embedding_model: str = "text-embedding-v4"
    openai_chat_model: str = "qwen3.5-35b-a3b"

    # ===== RAG 配置 =====
    rag_chunk_size: int = 2000           # 每个 chunk 的最大字符数
    rag_chunk_overlap: int = 400         # 相邻 chunk 的重叠字符数
    rag_vector_weight: float = 0.6       # 混合检索中向量检索的权重
    rag_fts_weight: float = 0.4          # 混合检索中全文检索的权重
    rag_rerank_model: str = "gte-rerank" # DashScope 重排序模型
    rag_rerank_top_k: int = 10           # 重排序前的候选数量
    rag_final_top_k: int = 5             # 重排序后的最终结果数量

    # ===== 去重配置 =====
    dedup_similarity_threshold: float = 0.85
```

### 配置说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rag_chunk_size` | 2000 | Chunk 大小，建议 1000-3000 |
| `rag_chunk_overlap` | 400 | 重叠大小，建议 chunk_size 的 10-20% |
| `rag_vector_weight` | 0.6 | 向量检索权重，与 fts_weight 之和应为 1 |
| `rag_fts_weight` | 0.4 | 全文检索权重 |
| `rag_rerank_top_k` | 10 | 重排序前候选数，建议 >= limit * 2 |
| `rag_final_top_k` | 5 | 最终返回结果数 |

---

## 7. 使用示例

### 7.1 索引新文章

```python
from app.services.chunking import get_chunking_service
from app.services.embedding import get_embedding_service
from app.services.vector_store import vector_store

async def index_article(session, article_id: uuid.UUID, title: str, description: str, content: str, summary: str):
    """索引文章内容。"""

    embedding_service = get_embedding_service()
    chunking_service = get_chunking_service()

    # 1. 创建 Summary Embedding
    summary_content = f"{title} {description}"
    summary_embedding = await embedding_service.embed_text(summary_content)
    await vector_store.store_embedding(
        session=session,
        article_id=article_id,
        embedding=summary_embedding,
        content_hash=hashlib.sha256(summary_content.encode()).hexdigest(),
    )

    # 2. 切分正文
    chunks = chunking_service.chunk_text_with_summary_first(
        text=content,
        summary=summary,
    )

    # 3. 批量生成嵌入
    chunk_texts = [c.text for c in chunks]
    chunk_embeddings = await embedding_service.embed_batch(chunk_texts)

    # 4. 存储 Chunk Embeddings
    chunk_data = [
        (c.text, c.start_char, c.end_char, emb)
        for c, emb in zip(chunks, chunk_embeddings)
    ]
    await vector_store.store_chunk_embeddings(
        session=session,
        article_id=article_id,
        chunks=chunk_data,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
    )
```

### 7.2 执行检索

```python
from app.services.embedding import get_embedding_service
from app.services.vector_store import vector_store
from app.services.reranker import get_reranker_service

async def search(session, query: str, limit: int = 5):
    """执行混合检索 + 重排序。"""

    # 1. 生成查询嵌入
    embedding_service = get_embedding_service()
    embedding = await embedding_service.embed_text(query)

    # 2. 混合检索
    results = await vector_store.hybrid_search(
        session=session,
        query=query,
        embedding=embedding,
        limit=10,  # 获取更多候选
    )

    # 3. 重排序
    reranker = get_reranker_service()
    ranked_results = await reranker.rerank(
        query=query,
        results=results,
        top_k=limit,
    )

    return ranked_results
```

### 7.3 Backfill 脚本

```bash
# 预览模式（不实际写入）
python scripts/backfill_embeddings.py --dry-run

# 执行数据预载
python scripts/backfill_embeddings.py --batch-size 10

# 只处理特定文章
python scripts/backfill_embeddings.py --article-id <uuid>
```

---

## 附录：文件清单

| 文件 | 用途 |
|------|------|
| `app/services/chunking.py` | 文本切分服务 |
| `app/services/embedding.py` | 向量嵌入服务 |
| `app/services/vector_store.py` | 向量存储与检索 |
| `app/services/reranker.py` | 重排序服务 |
| `app/services/query_transform.py` | 查询变换服务 |
| `app/services/compression.py` | 上下文压缩服务 |
| `app/models/orm_models.py` | ORM 模型定义 |
| `app/workflow/nodes.py` | Workflow 存储节点 |
| `app/deep_search/tools.py` | DeepSearch 工具 |
| `app/deep_search/prompts.py` | ReAct 提示词 |
| `app/deep_search/graph.py` | DeepSearch 编排 |
| `app/config.py` | 配置参数 |
| `alembic/versions/002_add_chunk_embeddings.py` | 数据库迁移 |
| `scripts/backfill_embeddings.py` | 数据预载脚本 |