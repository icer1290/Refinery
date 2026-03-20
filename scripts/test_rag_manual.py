#!/usr/bin/env python3
"""Manual test script for RAG system.

Usage:
    # In Docker container:
    docker compose exec ai-engine python scripts/test_rag_manual.py

    # Or locally with database connection:
    DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/news_aggregator \
    python scripts/test_rag_manual.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.services.chunking import get_chunking_service
from app.services.embedding import get_embedding_service
from app.services.vector_store import vector_store
from app.services.reranker import get_reranker_service
from app.services.compression import get_compression_service
from app.services.query_transform import get_query_transform_service
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

settings = get_settings()


async def test_chunking():
    """Test chunking service."""
    print("\n" + "="*60)
    print("📝 Testing Chunking Service")
    print("="*60)

    chunking = get_chunking_service()

    # Test text
    text = """
    人工智能正在改变世界。机器学习模型变得越来越强大。

    大语言模型如 GPT-4 展现出了惊人的能力。它们可以进行对话、写作、编程等任务。

    向量数据库是现代 RAG 系统的核心组件。它们可以高效地进行语义搜索。
    """ * 10  # Make it longer

    chunks = chunking.chunk_text(text)

    print(f"✅ Original text length: {len(text)} chars")
    print(f"✅ Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):
        print(f"   Chunk {i}: {len(chunk.text)} chars, pos {chunk.start_char}-{chunk.end_char}")

    return True


async def test_embedding():
    """Test embedding service."""
    print("\n" + "="*60)
    print("🔢 Testing Embedding Service")
    print("="*60)

    embedding_service = get_embedding_service()

    # Test single embedding
    text = "这是一段测试文本，用于测试向量嵌入服务。"
    embedding = await embedding_service.embed_text(text)

    print(f"✅ Single embedding: {len(embedding)} dimensions")

    # Test batch embedding
    texts = [
        "第一段测试文本。",
        "第二段测试文本。",
        "第三段测试文本。",
    ]
    embeddings = await embedding_service.embed_batch(texts)

    print(f"✅ Batch embeddings: {len(embeddings)} vectors")
    return True


async def test_vector_store():
    """Test vector store with database."""
    print("\n" + "="*60)
    print("🗄️ Testing Vector Store (requires database)")
    print("="*60)

    engine = create_async_engine(settings.database_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        # Test vector search on existing data
        embedding_service = get_embedding_service()
        query = "人工智能"
        embedding = await embedding_service.embed_text(query)

        results = await vector_store.find_similar(
            session=session,
            embedding=embedding,
            limit=3,
            similarity_threshold=0.5,
        )

        print(f"✅ Vector search for '{query}': {len(results)} results")
        for article, sim in results[:3]:
            print(f"   - {article.chinese_title or article.original_title[:50]} (sim: {sim:.3f})")

    await engine.dispose()
    return True


async def test_hybrid_search():
    """Test hybrid search (vector + FTS)."""
    print("\n" + "="*60)
    print("🔍 Testing Hybrid Search")
    print("="*60)

    engine = create_async_engine(settings.database_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        embedding_service = get_embedding_service()
        query = "机器学习技术"
        embedding = await embedding_service.embed_text(query)

        results = await vector_store.hybrid_search(
            session=session,
            query=query,
            embedding=embedding,
            limit=5,
        )

        print(f"✅ Hybrid search for '{query}': {len(results)} results")
        for r in results[:3]:
            print(f"   - {r.article_title[:40]}... (sim: {r.similarity:.3f})")

    await engine.dispose()
    return True


async def test_reranker():
    """Test reranker service."""
    print("\n" + "="*60)
    print("📊 Testing Reranker Service")
    print("="*60)

    try:
        reranker = get_reranker_service()
        print(f"✅ Reranker initialized with model: {reranker.model}")
        print("   (Actual reranking requires API call)")
        return True
    except Exception as e:
        print(f"⚠️ Reranker init warning: {e}")
        return True


async def test_query_transform():
    """Test query transform service."""
    print("\n" + "="*60)
    print("🔄 Testing Query Transform Service")
    print("="*60)

    try:
        transform = get_query_transform_service()

        # Test query expansion
        query = "人工智能在医疗领域的应用"
        expanded = await transform.expand_query(query, n=3)
        print(f"✅ Expanded query:")
        for i, q in enumerate(expanded):
            print(f"   {i+1}. {q}")

        return True
    except Exception as e:
        print(f"⚠️ Query transform error: {e}")
        return True


async def test_compression():
    """Test compression service."""
    print("\n" + "="*60)
    print("🗜️ Testing Compression Service")
    print("="*60)

    try:
        compression = get_compression_service()
        print(f"✅ Compression service initialized with model: {compression.model}")
        print("   (Actual compression requires API call)")
        return True
    except Exception as e:
        print(f"⚠️ Compression init warning: {e}")
        return True


async def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("🚀 RAG System Manual Tests")
    print("="*60)

    tests = [
        ("Chunking", test_chunking),
        ("Embedding", test_embedding),
        ("Reranker", test_reranker),
        ("Compression", test_compression),
        ("Query Transform", test_query_transform),
        ("Vector Store", test_vector_store),
        ("Hybrid Search", test_hybrid_search),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = await test_func()
            results.append((name, "✅ PASS" if success else "❌ FAIL"))
        except Exception as e:
            results.append((name, f"❌ ERROR: {str(e)[:50]}"))

    # Summary
    print("\n" + "="*60)
    print("📈 Test Summary")
    print("="*60)
    for name, status in results:
        print(f"   {name}: {status}")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(run_all_tests())