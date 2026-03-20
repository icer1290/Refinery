"""Prompt templates for GraphRAG workflows.

Prompts for:
- Entity extraction from articles
- Relationship extraction from articles
- Community summarization
- Deep graph analysis report generation
"""

from datetime import datetime

# === Entity Types ===

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


# === Entity Extraction Prompt ===

ENTITY_EXTRACTION_SYSTEM_PROMPT = """你是一个知识图谱实体提取专家。从新闻文章中识别和提取关键实体。

## 实体类型

{entity_types_desc}

## 输出格式

返回一个JSON对象，包含"entities"数组。每个实体包含：
- name: 实体名称（原文中的名称）
- type: 实体类型（PERSON, ORGANIZATION, TECHNOLOGY, PRODUCT, EVENT, LOCATION, CONCEPT）
- description: 简短描述（1-2句话）
- mentions: 文中出现该实体的文本片段（最多3个）
- confidence: 置信度（0.0-1.0）

## 提取原则

1. 只提取文章中明确提到的实体，不要推断
2. 优先提取与文章主题最相关的实体
3. 避免提取过于泛泛的概念（如"技术"、"公司"）
4. mentions应该包含实体出现的上下文
5. confidence反映实体在文章中的重要性

## 示例输出

{{
  "entities": [
    {{
      "name": "OpenAI",
      "type": "ORGANIZATION",
      "description": "人工智能研究实验室，ChatGPT的开发者",
      "mentions": ["OpenAI今日宣布推出新模型", "OpenAI的GPT-4模型"],
      "confidence": 0.95
    }},
    {{
      "name": "GPT-4",
      "type": "TECHNOLOGY",
      "description": "OpenAI开发的大型语言模型",
      "mentions": ["GPT-4在多项测试中表现出色"],
      "confidence": 0.9
    }}
  ]
}}

严格要求：
- 只返回JSON对象
- 不要使用Markdown代码块
- 不要添加解释性文字
"""

ENTITY_EXTRACTION_USER_PROMPT = """## 文章信息

标题: {title}
来源: {source}
发布时间: {published_at}

## 文章内容

{content}

## 任务

请从上述文章中提取关键实体，按JSON格式返回。"""


# === Relationship Extraction Prompt ===

RELATIONSHIP_EXTRACTION_SYSTEM_PROMPT = """你是一个知识图谱关系提取专家。从新闻文章中识别实体之间的关系。

## 关系类型示例

常见的实体关系类型：
- develops: 开发（如：OpenAI develops GPT-4）
- acquires: 收购（如：Microsoft acquires GitHub）
- competes_with: 竞争（如：Google competes_with OpenAI）
- partners_with: 合作（如：Apple partners_with OpenAI）
- uses: 使用（如：Tesla uses NVIDIA chips）
- invests_in: 投资（如：Sequoia invests_in OpenAI）
- leads: 领导（如：Sam Altman leads OpenAI）
- launches: 发布（如：Apple launches Vision Pro）
- works_on: 研发（如：DeepMind works_on AlphaFold）
- member_of: 属于（如：GPT-4 member_of LLM family）

## 输出格式

返回一个JSON对象，包含"relationships"数组。每个关系包含：
- source_entity: 源实体名称
- target_entity: 目标实体名称
- relation_type: 关系类型
- description: 关系描述（1句话）
- evidence: 支持该关系的原文片段

## 提取原则

1. 只提取文章中明确描述的关系
2. 关系应该是具体的、有意义的
3. source_entity和target_entity必须在文章中被提及
4. evidence必须来自原文

## 示例输出

{{
  "relationships": [
    {{
      "source_entity": "OpenAI",
      "target_entity": "GPT-4",
      "relation_type": "develops",
      "description": "OpenAI开发了GPT-4语言模型",
      "evidence": "OpenAI今日宣布GPT-4正式发布"
    }},
    {{
      "source_entity": "Microsoft",
      "target_entity": "OpenAI",
      "relation_type": "invests_in",
      "description": "微软向OpenAI投资数十亿美元",
      "evidence": "微软宣布向OpenAI追加100亿美元投资"
    }}
  ]
}}

严格要求：
- 只返回JSON对象
- 不要使用Markdown代码块
- 不要添加解释性文字
"""

RELATIONSHIP_EXTRACTION_USER_PROMPT = """## 文章信息

标题: {title}
来源: {source}

## 文章内容

{content}

## 已识别的实体

{entities}

## 任务

请从上述文章中提取实体之间的关系，按JSON格式返回。"""


# === Community Summarization Prompt ===

COMMUNITY_SUMMARY_PROMPT = """你是一个知识图谱分析专家。请为一个实体社区生成摘要。

## 社区信息

社区名称: {community_name}
实体列表: {entity_list}

## 实体详情

{entity_details}

## 任务

请为这个社区生成：
1. 一个简短的社区名称（2-5个词）
2. 一个社区摘要（2-3句话，描述这些实体为何被归类在一起）

## 输出格式

返回JSON对象：
{{
  "name": "社区名称",
  "summary": "社区摘要描述"
}}

严格要求：
- 只返回JSON对象
- 不要使用Markdown代码块
"""


# === Deep Graph Analysis Report Prompt ===

DEEP_GRAPH_REPORT_PROMPT = """你是一位资深科技行业分析师，请基于知识图谱生成深度分析报告。

## 重要：当前时间参考

**当前日期和时间：{current_time}**
请以此作为"现在"的参考点分析新闻的时效性。

## 选定文章信息

{articles_info}

## 知识图谱信息

### 关键实体 ({entity_count}个)

{entities_info}

### 实体关系 ({relationship_count}条)

{relationships_info}

### 社区分析 ({community_count}个)

{communities_info}

## 扩展实体信息

以下是通过图扩展发现的关联实体：

{expanded_entities_info}

## 报告要求

请生成一份结构化的深度分析报告，包含以下部分：

### 1. 执行摘要
- 用3-5句话概括核心发现
- 突出最重要的洞察

### 2. 关键实体分析
- 分析最重要的3-5个实体
- 说明它们在新闻中的作用和关联
- 标注扩展发现的关联实体

### 3. 关系网络洞察
- 分析实体间的关系模式
- 识别关键枢纽节点
- 发现跨文章的关联线索

### 4. 社区分析
- 分析各社区的共同特征
- 解释社区间的关联
- 识别潜在的合作或竞争关系

### 5. 行业趋势识别
- 基于图谱分析行业动态
- 识别新兴趋势或变化
- 预测可能的发展方向

### 6. 跨文章关联发现
- 分析不同文章间的隐含联系
- 识别共享的关键实体
- 发现一致或矛盾的报道

### 7. 后续关注建议
- 值得关注的后续发展
- 建议深入了解的方向

## 输出格式

请以中文撰写，使用专业但易懂的语言。每个部分用标题分隔，内容简洁有力。
使用 **加粗** 标注重要的实体名称和关键发现。
"""


# === Helper Functions ===

def format_entity_types() -> str:
    """Format entity types for prompts."""
    lines = []
    for etype, desc in ENTITY_TYPE_DESCRIPTIONS.items():
        lines.append(f"- {etype}: {desc}")
    return "\n".join(lines)


def get_entity_extraction_prompts() -> tuple[str, str]:
    """Get entity extraction prompts.

    Returns:
        Tuple of (system_prompt, user_prompt_template)
    """
    system_prompt = ENTITY_EXTRACTION_SYSTEM_PROMPT.format(
        entity_types_desc=format_entity_types()
    )
    return system_prompt, ENTITY_EXTRACTION_USER_PROMPT


def get_relationship_extraction_prompts() -> tuple[str, str]:
    """Get relationship extraction prompts.

    Returns:
        Tuple of (system_prompt, user_prompt_template)
    """
    return RELATIONSHIP_EXTRACTION_SYSTEM_PROMPT, RELATIONSHIP_EXTRACTION_USER_PROMPT


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


def format_graph_for_report(
    entities: list,
    relationships: list,
    communities: list,
    expanded_entities: list | None = None,
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