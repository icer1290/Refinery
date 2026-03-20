"""Centralized prompt management system.

This module provides:
- PromptTemplate dataclass for type-safe prompt templates
- A registry of all prompts used in the application
- Helper functions for formatting prompt data

Usage:
    from app.prompts import get_prompt, PromptTemplate

    prompt = get_prompt("scoring.article")
    formatted = prompt.format(title="...", description="...", content_section="...")
"""

from app.prompts.templates import PromptTemplate

# === Prompt Registry ===

_PROMPTS: dict[str, PromptTemplate] = {}


def register_prompt(prompt: PromptTemplate) -> None:
    """Register a prompt template.

    Args:
        prompt: PromptTemplate instance to register
    """
    if prompt.id in _PROMPTS:
        raise ValueError(f"Prompt with id '{prompt.id}' already registered")
    _PROMPTS[prompt.id] = prompt


def get_prompt(prompt_id: str) -> PromptTemplate:
    """Get a prompt template by ID.

    Args:
        prompt_id: The unique prompt identifier

    Returns:
        PromptTemplate instance

    Raises:
        KeyError: If prompt not found
    """
    if prompt_id not in _PROMPTS:
        raise KeyError(f"Prompt '{prompt_id}' not found in registry")
    return _PROMPTS[prompt_id]


def list_prompts() -> list[str]:
    """List all registered prompt IDs.

    Returns:
        List of prompt IDs
    """
    return list(_PROMPTS.keys())


# ============================================================================
# SCORING PROMPTS
# ============================================================================

register_prompt(
    PromptTemplate(
        id="scoring.article",
        template="""你是一位科技新闻编辑专家。请从以下三个维度对文章进行评分(0-10分):

1. 行业影响力 (industry_impact_score): 对科技行业的影响力
   - 是否涉及重大技术突破或产品发布
   - 是否影响行业格局或市场趋势
   - 对开发者和企业的影响程度

2. 关键节点 (milestone_score): 事件的里程碑意义
   - 是否代表技术发展的重要节点
   - 是否开创先例或改变现状
   - 历史意义和长期价值

3. 引人关注 (attention_score): 新闻价值
   - 受众广泛程度
   - 话题热度和传播潜力
   - 时效性和独特性

文章标题: {title}
文章描述: {description}{content_section}

请以JSON格式返回评分结果:
{{
    "industry_impact_score": <分数>,
    "milestone_score": <分数>,
    "attention_score": <分数>,
    "reasoning": "<评分理由简述>"
}}

只返回JSON，不要添加其他内容。""",
        variables=["title", "description", "content_section"],
        description="Article scoring prompt for evaluating industry impact, milestone, and attention",
        version="1.0.0",
    )
)


# ============================================================================
# TRANSLATION PROMPTS
# ============================================================================

register_prompt(
    PromptTemplate(
        id="translation.article",
        template="""## Role
你是一位冷静、犀利且极具洞察力的科技新闻资深编辑，擅长从复杂的工程文档和战略报告中榨取核心价值。

## Task
请基于提供的文章全文，撰写一段极简、高信息密度的中文专业简报。

## Output Format (Strict Requirements)
- `chinese_title`：必须是 `[领域] 标题` 格式
  - 示例: `[AI] GPT-5发布`, `[创业公司] Cluely CEO承认虚报年收入`
  - 不合格: `GPT-5发布` (缺少领域标签)

- `chinese_summary`：必须是一个普通字符串，不要拆成额外 JSON 字段；内容由 3 段组成，用空行分隔：
  1. 第一段：`动作 + 关键结果`，随后用一句话点出最核心的行业变量。
  2. 第二段：2-3 个要点列出核心数据、技术参数或战略动作。**每个要点必须以 `· ` 开头**
  3. 第三段：以 `主编洞察：` 开头，用 3-5 句话点破其对行业格局、竞品逻辑或未来演进的深层影响。

## Style & Rules
- **禁止额外字段**：JSON 只允许 `chinese_title`、`chinese_summary`、`entities_preserved` 三个键，严禁输出 `第一段 (核心内容)`、`第二段 (关键事实)`、markdown 编号或其他键。
- **禁止标签**：严禁在摘要正文中出现"标题"、"钩子"、"核心事实"、"深度洞察"等指示性标签。
- **去除废话**：严禁使用"据悉"、"令人震惊"、"本文介绍了"、"...具有里程碑意义"等公关词汇或AI腔。
- **数据驱动**：原文中的数字、百分比、融资金额、技术参数必须精准保留在事实要点中。
- **冷静权威**：语气应像给顶级决策者看的简报，保持客观、克制且具有穿透力。
- **篇幅限制**：全文严格控制在 300 字以内。
{entities_section}{feedback_section}
## Example Output
```json
{{
    "chinese_title": "[创业公司] Cluely CEO承认虚报年收入",
    "chinese_summary": "Cluely创始人兼CEO Roy Lee在X平台公开承认，去年向TechCrunch披露的700万美元年经常性收入（ARR）为虚假数据。此举暴露其早期依赖炒作驱动增长的模式已难以为继，行业对初创企业可信度的审查正加速收紧。\\n\\n· 2025年夏季，Cluely宣称拥有700万美元ARR，后被证实为虚构；实际营收数据来自其Stripe账户，未公开具体金额。\\n· 公司于2025年6月完成1500万美元Series A融资，由Andreessen Horowitz领投，此前已获530万美元种子轮融资。\\n· 该公司原定位为"面试作弊工具"，现转型为AI会议笔记产品，但其营销策略仍以病毒式争议为核心。\\n\\n主编洞察：虚报数据并非孤立事件，而是典型"现象级初创"在缺乏可持续商业模式时的生存策略。当舆论热度退潮，真实财务表现成为唯一检验标准。该事件将倒逼风投机构强化尽调维度，从"传播力"转向"现金流健康度"。未来，所有以情绪驱动增长的AI应用都将面临更严苛的合规与透明度压力。",
    "entities_preserved": ["Cluely", "Roy Lee", "TechCrunch", "Andreessen Horowitz", "Stripe"]
}}
```

---

**文章标题**: {title}

**文章内容**:
{content}

请以JSON格式返回:
{{
    "chinese_title": "<标题，即[领域] + 标题>",
    "chinese_summary": "<完整简报内容，包含三段>",
    "entities_preserved": ["<保留原文的实体列表>"]
}}

只返回JSON，不要添加其他内容。""",
        variables=["title", "content", "entities_section", "feedback_section"],
        description="Article translation and summarization prompt with Chinese output format",
        version="1.0.0",
    )
)


register_prompt(
    PromptTemplate(
        id="reflection.check",
        template="""请检查以下翻译是否符合要求:

原文标题: {original_title}
原文内容片段: {original_content_preview}

中文标题: {chinese_title}
中文摘要: {chinese_summary}

## 检查项目（必须严格检查以下三项格式要求）

### 1. 标题格式检查
- 必须以 `[领域]` 开头，领域标签后跟空格和标题
- 合格示例: `[AI] GPT-5发布`, `[创业公司] Cluely CEO承认虚报年收入`
- 不合格示例: `GPT-5发布` (缺少标签), `[AI]GPT-5发布` (缺少空格)

### 2. 摘要结构检查
- 必须包含三个段落，用空行分隔
- 第一段：动作 + 关键结果（无需特殊前缀）
- 第二段：必须包含 2-3 个要点，**每个要点必须以 `· ` 开头**
- 第三段：必须以 `主编洞察：` 开头
- 不合格示例：第二段使用 `- ` 或 `* ` 或数字编号，第三段缺少 `主编洞察：` 前缀

### 3. 实体保留检查
- 重要人名、公司名、产品名是否正确处理
- 未出现错误的中文翻译或遗漏

## 判定标准
- 三项全部通过: passed = true
- 任一项不通过: passed = false，并在 issues 中列出具体问题

请以JSON格式返回检查结果:
{{
    "passed": <true或false>,
    "issues": ["<问题列表，如: '标题格式错误：缺少领域标签'", "第二段格式错误：要点未以·开头">"],
    "feedback": "<改进建议，如果没有问题则为null>"
}}

只返回JSON，不要添加其他内容。""",
        variables=["original_title", "original_content_preview", "chinese_title", "chinese_summary"],
        description="Translation quality reflection check prompt",
        version="1.0.0",
    )
)


# ============================================================================
# DEEP SEARCH PROMPTS
# ============================================================================

register_prompt(
    PromptTemplate(
        id="deep_search.react_system",
        template="""你是一个深度新闻分析助手，使用 ReAct (Reasoning + Acting) 方法来收集新闻的背景信息和前因后果。

## 重要：当前时间参考

**当前日期和时间：{current_time}**
请以此作为"现在"的参考点。当分析新闻发布时间时：
- 早于此时间的事件为"过去"
- 晚于此时间的事件为"未来"或"预测性内容"

## 可用工具

1. **vector_search** - 在本地数据库中搜索相关文章（优先使用）
   - 输入: {{"query": "搜索查询", "limit": 5}}
   - 用途: 查找历史相关报道、背景文章、技术细节
   - 注意: 本地数据库包含详细的文章全文，适合查找技术背景和历史脉络

2. **web_search** - 在网络上搜索相关信息
   - 输入: {{"query": "搜索查询"}}
   - 用途: 获取最新外部信息、官方声明、实时新闻

## 工作流程

1. **优先本地**: 先使用 vector_search 查找本地相关文章
2. **补充外部**: 如果本地信息不足，再使用 web_search
3. **思考 (Thought)**: 分析当前信息，决定下一步行动
4. **行动 (Action)**: 选择工具并执行
5. **观察 (Observation)**: 分析工具返回的结果
6. 重复直到收集足够信息

## 输出格式

每次回复必须是一个JSON对象，包含以下字段:
- thought: 你的思考过程
- action: "vector_search" 或 "web_search" 或 "conclude"
- action_input: 工具输入（如果是conclude则为null）

严格要求:
- 只返回 JSON 对象本身
- 不要使用 ```json 或其他 Markdown 包裹
- 不要在 JSON 前后添加解释性文字
- 如果信息不足，请继续搜索，不要输出半截 JSON

## 示例

{{"thought": "文章提到OpenAI发布新模型，我先在本地数据库搜索相关历史报道", "action": "vector_search", "action_input": {{"query": "OpenAI 模型发布", "limit": 5}}}}

{{"thought": "本地数据库信息不够，需要搜索网络上的最新报道", "action": "web_search", "action_input": {{"query": "OpenAI 最新模型发布 2026"}}}}

{{"thought": "已经收集了足够的背景信息，可以生成报告了", "action": "conclude", "action_input": null}}
""",
        variables=["current_time"],
        description="ReAct system prompt for deep search workflow",
        version="1.0.0",
    )
)


register_prompt(
    PromptTemplate(
        id="deep_search.react_user",
        template="""## 原始文章

标题: {title}
来源: {source}
发布时间: {published_at}

内容摘要:
{summary}

## 已收集信息

{collected_info}

## 当前状态

迭代: {current_iteration}/{max_iterations}
工具调用历史: {tool_count} 次

## 下一步

请分析当前信息，决定下一步行动。如果已有足够信息，请选择 "conclude" 生成报告。
""",
        variables=["title", "source", "published_at", "summary", "collected_info", "current_iteration", "max_iterations", "tool_count"],
        description="ReAct user prompt for deep search workflow",
        version="1.0.0",
    )
)


register_prompt(
    PromptTemplate(
        id="deep_search.conclusion",
        template="""你是一位资深科技新闻编辑，请基于收集的信息撰写一份深度追踪报告。

## 重要：当前时间参考

**当前日期和时间：{current_time}**
请以此作为"现在"的参考点分析新闻的时效性。

## 原始文章

标题: {title}
来源: {source}
发布时间: {published_at}

内容摘要:
{summary}

## 收集的背景信息

{collected_info}

## 报告要求

请生成一份结构化的深度追踪报告，包含以下部分：

### 1. 事件概述
- 用2-3句话概括新闻核心事件
- 突出关键事实和数据

### 2. 背景信息
- 相关技术/公司/行业背景
- 历史发展脉络

### 3. 相关历史
- 类似事件或相关报道
- 时间线和因果关系

### 4. 行业影响分析
- 对行业格局的影响
- 对竞争对手的影响
- 对开发者/用户的影响

### 5. 后续关注点
- 值得关注的后续发展
- 可能的衍生新闻

## 输出格式

请以中文撰写，使用专业但易懂的语言。每个部分用标题分隔，内容简洁有力。
""",
        variables=["current_time", "title", "source", "published_at", "summary", "collected_info"],
        description="Conclusion prompt for generating deep search reports",
        version="1.0.0",
    )
)


register_prompt(
    PromptTemplate(
        id="deep_search.json_repair",
        template="""你上一个回复未能被解析为合法 JSON。

请严格返回一个 JSON 对象，不要使用 Markdown 代码块，不要补充解释，不要截断。

输出格式:
{"thought": "<string>", "action": "vector_search|web_search|conclude", "action_input": <object|null>}
""",
        variables=[],
        description="Prompt for repairing malformed JSON responses in deep search",
        version="1.0.0",
    )
)


# ============================================================================
# DEEP GRAPH PROMPTS
# ============================================================================

register_prompt(
    PromptTemplate(
        id="deep_graph.entity_extraction_system",
        template="""你是一个知识图谱实体提取专家。从新闻文章中识别和提取关键实体。

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
""",
        variables=["entity_types_desc"],
        description="System prompt for entity extraction in GraphRAG",
        version="1.0.0",
    )
)


register_prompt(
    PromptTemplate(
        id="deep_graph.entity_extraction_user",
        template="""## 文章信息

标题: {title}
来源: {source}
发布时间: {published_at}

## 文章内容

{content}

## 任务

请从上述文章中提取关键实体，按JSON格式返回。""",
        variables=["title", "source", "published_at", "content"],
        description="User prompt for entity extraction in GraphRAG",
        version="1.0.0",
    )
)


register_prompt(
    PromptTemplate(
        id="deep_graph.relationship_extraction_system",
        template="""你是一个知识图谱关系提取专家。从新闻文章中识别实体之间的关系。

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
""",
        variables=[],
        description="System prompt for relationship extraction in GraphRAG",
        version="1.0.0",
    )
)


register_prompt(
    PromptTemplate(
        id="deep_graph.relationship_extraction_user",
        template="""## 文章信息

标题: {title}
来源: {source}

## 文章内容

{content}

## 已识别的实体

{entities}

## 任务

请从上述文章中提取实体之间的关系，按JSON格式返回。""",
        variables=["title", "source", "content", "entities"],
        description="User prompt for relationship extraction in GraphRAG",
        version="1.0.0",
    )
)


register_prompt(
    PromptTemplate(
        id="deep_graph.community_summary",
        template="""你是一个知识图谱分析专家。请为一个实体社区生成摘要。

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
""",
        variables=["community_name", "entity_list", "entity_details"],
        description="Prompt for community summarization in GraphRAG",
        version="1.0.0",
    )
)


register_prompt(
    PromptTemplate(
        id="deep_graph.report",
        template="""你是一位资深科技行业分析师，请基于知识图谱生成深度分析报告。

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
""",
        variables=["current_time", "articles_info", "entity_count", "entities_info", "relationship_count", "relationships_info", "community_count", "communities_info", "expanded_entities_info"],
        description="Prompt for generating deep graph analysis reports",
        version="1.0.0",
    )
)


# ============================================================================
# RAG PROMPTS
# ============================================================================

register_prompt(
    PromptTemplate(
        id="rag.hyde",
        template="""请生成一篇假设性的新闻文章，这篇文章能够完美回答以下问题。
文章应该包含详细的技术细节和背景信息。
目标长度约{doc_length}字。

问题: {query}

请直接输出文章内容，不要添加任何解释或标注:""",
        variables=["query", "doc_length"],
        description="HyDE (Hypothetical Document Embedding) prompt for generating hypothetical documents",
        version="1.0.0",
    )
)


register_prompt(
    PromptTemplate(
        id="rag.query_expand",
        template="""你是一个搜索助手。请将以下查询扩展为{n}个相关但不同的搜索查询。
这些查询应该从不同角度探索原始问题的不同方面。
每个查询应该是一个独立的问题，可以单独用于搜索。

原始查询: {query}

请以JSON数组格式输出，例如:
["查询1", "查询2", "查询3"]

只输出JSON数组，不要添加其他内容:""",
        variables=["query", "n"],
        description="Multi-query expansion prompt for improving recall",
        version="1.0.0",
    )
)


register_prompt(
    PromptTemplate(
        id="rag.keyword_extract",
        template="""从以下查询中提取{n}个最重要的关键词。
关键词应该是技术术语、公司名称、产品名称或核心概念。

查询: {query}

请以JSON数组格式输出，例如:
["关键词1", "关键词2", "关键词3"]

只输出JSON数组:""",
        variables=["query", "n"],
        description="Keyword extraction prompt for improving full-text search",
        version="1.0.0",
    )
)


# ============================================================================
# COMPRESSION PROMPTS
# ============================================================================

register_prompt(
    PromptTemplate(
        id="compression.chunk",
        template="""你是一个信息提取助手。请从以下多个文档片段中提取与问题直接相关的信息。

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

请输出压缩后的相关内容（不超过{max_length}字）:""",
        variables=["query", "chunks_text", "max_length"],
        description="Chunk compression prompt for extracting relevant information",
        version="1.0.0",
    )
)


register_prompt(
    PromptTemplate(
        id="compression.key_sentences",
        template="""从以下文档中提取{max_sentences}个最相关的问题答案的句子。
按相关性排序输出。

问题: {query}

文档:
{chunks_text}

请以JSON数组格式输出，每个元素包含sentence和source字段:
[{{"sentence": "相关句子", "source": "来源名称"}}, ...]

只输出JSON数组:""",
        variables=["query", "chunks_text", "max_sentences"],
        description="Key sentence extraction prompt for lightweight context extraction",
        version="1.0.0",
    )
)


register_prompt(
    PromptTemplate(
        id="compression.context_summary",
        template="""请根据以下资料，为回答问题准备一份简洁的背景摘要。
摘要应该:
1. 整合多个来源的信息
2. 突出与问题最相关的事实和数据
3. 保持客观，准确反映原始资料
4. 长度控制在500字以内

问题: {query}

资料:
{chunks_text}

背景摘要:""",
        variables=["query", "chunks_text"],
        description="Context summary prompt for generating query-focused summaries",
        version="1.0.0",
    )
)


# Re-export for convenience
__all__ = [
    "PromptTemplate",
    "get_prompt",
    "register_prompt",
    "list_prompts",
]