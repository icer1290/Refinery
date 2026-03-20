"""Helper functions for prompt formatting.

Contains utility functions extracted from deep_graph/prompts.py and
other modules that format data for prompt templates.
"""

from datetime import datetime
from typing import Optional


# === Entity Types (from deep_graph/prompts.py) ===

ENTITY_TYPES = [
    "PERSON",        # People (e.g., "Sam Altman", "Elon Musk")
    "ORGANIZATION",  # Companies, institutions (e.g., "OpenAI", "Google")
    "TECHNOLOGY",    # Technologies, frameworks (e.g., "GPT-4", "Transformer")
    "PRODUCT",       # Products, services (e.g., "ChatGPT", "Gemini")
    "EVENT",         # Events, conferences (e.g., "WWDC 2026", "AI Summit")
    "LOCATION",      # Places (e.g., "Silicon Valley", "China")
    "CONCEPT",       # Concepts, ideas (e.g., "Artificial Intelligence", "Machine Learning")
]

ENTITY_TYPE_DESCRIPTIONS = {
    "PERSON": "人物 - 公司高管、研究人员、开发者等",
    "ORGANIZATION": "组织 - 公司、研究机构、开源项目等",
    "TECHNOLOGY": "技术 - AI模型、框架、算法、协议等",
    "PRODUCT": "产品 - 应用、服务、硬件产品等",
    "EVENT": "事件 - 发布会、会议、里程碑事件等",
    "LOCATION": "地点 - 城市、国家、地区等",
    "CONCEPT": "概念 - 技术概念、方法论、理念等",
}


def format_entity_types() -> str:
    """Format entity types for prompts.

    Returns:
        Formatted string listing all entity types with descriptions.
    """
    lines = []
    for etype, desc in ENTITY_TYPE_DESCRIPTIONS.items():
        lines.append(f"- {etype}: {desc}")
    return "\n".join(lines)


def format_entities_for_prompt(entities: list) -> str:
    """Format entity list for relationship extraction prompt.

    Args:
        entities: List of extracted entities

    Returns:
        Formatted string
    """
    if not entities:
        return "无已识别实体"

    lines = []
    for i, entity in enumerate(entities, 1):
        lines.append(f"{i}. {entity['name']} ({entity['type']})")
        if entity.get('description'):
            lines.append(f"   描述: {entity['description']}")

    return "\n".join(lines)


def format_collected_info(collected_info: list) -> str:
    """Format collected info for deep search prompts.

    Args:
        collected_info: List of collected information

    Returns:
        Formatted string
    """
    if not collected_info:
        return "暂无收集信息"

    formatted = []
    for i, info in enumerate(collected_info, 1):
        source = info.get("source", "unknown")
        content = info.get("content", "")
        relevance = info.get("relevance", "")
        formatted.append(f"### 信息 {i} (来源: {source})")
        formatted.append(f"相关性: {relevance}")
        formatted.append(f"内容: {content[:500]}...")
        formatted.append("")

    return "\n".join(formatted)


def format_graph_for_report(
    entities: list,
    relationships: list,
    communities: list,
    expanded_entities: Optional[list] = None,
) -> dict[str, str]:
    """Format graph data for report prompt.

    Args:
        entities: List of graph nodes
        relationships: List of graph edges
        communities: List of community data
        expanded_entities: List of expanded entity context

    Returns:
        Dict with formatted strings for each section
    """
    # Format entities
    entities_info = []
    for entity in entities[:20]:  # Limit to top 20
        marker = "📌 " if entity.get("is_expanded") else ""
        entities_info.append(
            f"- {marker}{entity['label']} ({entity['type']}): {entity.get('description', 'N/A')}"
        )
    entities_str = "\n".join(entities_info) if entities_info else "无实体信息"

    # Format relationships
    rel_info = []
    for rel in relationships[:20]:  # Limit to top 20
        marker = "🔗 " if rel.get("is_expanded") else ""
        rel_info.append(
            f"- {marker}{rel['source']} --[{rel['relation_type']}]--> {rel['target']}"
        )
    rel_str = "\n".join(rel_info) if rel_info else "无关系信息"

    # Format communities
    comm_info = []
    for comm in communities[:10]:  # Limit to top 10
        comm_info.append(
            f"- {comm['name']}: {comm.get('summary', 'N/A')} ({comm['entity_count']}个实体)"
        )
    comm_str = "\n".join(comm_info) if comm_info else "无社区信息"

    # Format expanded entities
    if expanded_entities:
        exp_info = []
        for exp in expanded_entities[:10]:
            exp_info.append(
                f"- {exp.get('entity_id', 'Unknown')}: 相关度={exp.get('relevance_score', 0):.2f}, "
                f"跳数={exp.get('hop_distance', 0)}"
            )
        exp_str = "\n".join(exp_info)
    else:
        exp_str = "无扩展实体"

    return {
        "entities_info": entities_str,
        "relationships_info": rel_str,
        "communities_info": comm_str,
        "expanded_entities_info": exp_str,
        "entity_count": str(len(entities)),
        "relationship_count": str(len(relationships)),
        "community_count": str(len(communities)),
    }


def format_articles_for_report(articles: list) -> str:
    """Format article list for report prompt.

    Args:
        articles: List of article dicts

    Returns:
        Formatted string
    """
    if not articles:
        return "无选定文章"

    lines = []
    for i, article in enumerate(articles, 1):
        lines.append(f"### 文章 {i}")
        lines.append(f"标题: {article.get('title', 'N/A')}")
        lines.append(f"来源: {article.get('source', 'N/A')}")
        lines.append(f"发布时间: {article.get('published_at', 'N/A')}")
        if article.get('summary'):
            lines.append(f"摘要: {article['summary'][:200]}...")
        lines.append("")

    return "\n".join(lines)


def get_current_time() -> str:
    """Get current time formatted for prompts.

    Returns:
        Current time string in Chinese format.
    """
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")


def build_content_section(content: Optional[str], max_length: int = 2000) -> str:
    """Build content section for scoring prompt.

    Args:
        content: Article content (optional)
        max_length: Maximum content length

    Returns:
        Formatted content section string
    """
    if not content:
        return ""
    return f"\n\nContent:\n{content[:max_length]}"


def build_entities_section(entities_to_preserve: Optional[list[str]]) -> str:
    """Build entities preservation section for translation prompt.

    Args:
        entities_to_preserve: List of entities to keep in original language

    Returns:
        Formatted entities section string
    """
    if not entities_to_preserve:
        return ""
    return f"\n\n**实体保留要求**：以下专有名词请保留原文（人名/公司名/产品名）：{', '.join(entities_to_preserve)}"


def build_feedback_section(feedback: Optional[str]) -> str:
    """Build feedback section for translation retry.

    Args:
        feedback: Feedback from previous reflection

    Returns:
        Formatted feedback section string
    """
    if not feedback:
        return ""
    return f"""

## 重要修正要求
上一次翻译存在以下问题，必须修正:
{feedback}
"""