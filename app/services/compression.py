"""Context compression service for RAG system.

Uses LLM to extract and compress relevant information from retrieved chunks,
reducing noise and improving the quality of context provided to the generator.
"""

from typing import List

from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.core import get_logger
from app.services.vector_store import SearchResult

logger = get_logger(__name__)
settings = get_settings()


class CompressionService:
    """Service for compressing retrieved context using LLM.

    Compresses multiple chunks into a focused, relevant passage
    that directly addresses the query.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        """Initialize the compression service.

        Args:
            model: Chat model name (default from config)
            api_key: API key (default from settings)
            base_url: API base URL (default from settings)
        """
        self.model = model or settings.openai_chat_model
        self.api_key = api_key or settings.openai_api_key
        self.base_url = base_url or settings.openai_base_url

        # Initialize LLM with low temperature for extraction
        self.llm = ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=0.1,
        )

        logger.info(
            "CompressionService initialized",
            model=self.model,
        )

    async def compress_chunks(
        self,
        query: str,
        results: List[SearchResult],
        max_length: int = 2000,
    ) -> str:
        """Compress multiple chunks into a focused context.

        Extracts only the information relevant to the query from
        multiple chunks and combines them into a coherent passage.

        Args:
            query: The search query
            results: List of search results to compress
            max_length: Maximum length of compressed context

        Returns:
            Compressed context string
        """
        if not results:
            return ""

        try:
            # Combine all chunk texts with source markers
            chunks_text = "\n\n---\n\n".join(
                f"[来源: {r.source_name}]\n{r.chunk_text}"
                for r in results
            )

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

请输出压缩后的相关内容（不超过{max_length}字）:"""

            response = await self.llm.ainvoke(prompt)
            compressed = response.content

            # Truncate if still too long
            if len(compressed) > max_length:
                compressed = compressed[:max_length].rsplit("。", 1)[0] + "..."

            logger.info(
                "Compressed chunks",
                query=query[:50],
                original_length=len(chunks_text),
                compressed_length=len(compressed),
                num_chunks=len(results),
            )

            return compressed

        except Exception as e:
            logger.error(
                "Failed to compress chunks",
                error=str(e),
                query=query[:50],
            )
            # Fallback: return concatenated chunks
            combined = "\n\n".join(r.chunk_text for r in results[:3])
            return combined[:max_length]

    async def extract_key_sentences(
        self,
        query: str,
        results: List[SearchResult],
        max_sentences: int = 10,
    ) -> List[str]:
        """Extract key sentences from chunks that answer the query.

        A lighter-weight alternative to full compression that
        extracts individual relevant sentences.

        Args:
            query: The search query
            results: List of search results
            max_sentences: Maximum number of sentences to extract

        Returns:
            List of relevant sentences with source attribution
        """
        if not results:
            return []

        try:
            chunks_text = "\n\n---\n\n".join(
                f"[{r.source_name}] {r.chunk_text}"
                for r in results
            )

            prompt = f"""从以下文档中提取{max_sentences}个最相关的问题答案的句子。
按相关性排序输出。

问题: {query}

文档:
{chunks_text}

请以JSON数组格式输出，每个元素包含sentence和source字段:
[{{"sentence": "相关句子", "source": "来源名称"}}, ...]

只输出JSON数组:"""

            response = await self.llm.ainvoke(prompt)
            content = response.content.strip()

            import json
            # Handle potential markdown code blocks
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content.rsplit("```", 1)[0]
                content = content.strip()

            start = content.find("[")
            end = content.rfind("]")
            if start != -1 and end != -1:
                json_str = content[start:end + 1]
                extracted = json.loads(json_str)
                sentences = [
                    f"{item['sentence']} ({item.get('source', '未知')})"
                    for item in extracted[:max_sentences]
                ]
            else:
                sentences = []

            logger.debug(
                "Extracted key sentences",
                query=query[:50],
                num_sentences=len(sentences),
            )

            return sentences

        except Exception as e:
            logger.error(
                "Failed to extract key sentences",
                error=str(e),
                query=query[:50],
            )
            return []

    async def summarize_for_context(
        self,
        query: str,
        results: List[SearchResult],
    ) -> str:
        """Create a summary specifically for answering the query.

        Generates a summary that focuses on information needed
        to answer the query, suitable for use as LLM context.

        Args:
            query: The search query
            results: List of search results

        Returns:
            Query-focused summary
        """
        if not results:
            return ""

        try:
            # Sort results by similarity
            sorted_results = sorted(results, key=lambda r: r.similarity, reverse=True)

            # Take top chunks
            top_chunks = sorted_results[:5]

            chunks_text = "\n\n".join(
                f"【{r.source_name}】\n{r.chunk_text}"
                for r in top_chunks
            )

            prompt = f"""请根据以下资料，为回答问题准备一份简洁的背景摘要。
摘要应该:
1. 整合多个来源的信息
2. 突出与问题最相关的事实和数据
3. 保持客观，准确反映原始资料
4. 长度控制在500字以内

问题: {query}

资料:
{chunks_text}

背景摘要:"""

            response = await self.llm.ainvoke(prompt)
            summary = response.content

            logger.info(
                "Generated context summary",
                query=query[:50],
                summary_length=len(summary),
            )

            return summary

        except Exception as e:
            logger.error(
                "Failed to generate context summary",
                error=str(e),
                query=query[:50],
            )
            return "\n\n".join(r.chunk_text for r in results[:3])


# Singleton instance
_compression_service: CompressionService | None = None


def get_compression_service() -> CompressionService:
    """Get or create compression service instance."""
    global _compression_service
    if _compression_service is None:
        _compression_service = CompressionService()
    return _compression_service