"""RSS feed source constants."""

from typing import NamedTuple


class RSSFeed(NamedTuple):
    """RSS feed source definition."""

    name: str
    url: str
    category: str


# Default RSS feed sources
DEFAULT_RSS_FEEDS: list[RSSFeed] = [
    # Tech News
    RSSFeed(
        name="Ars Technica",
        url="https://feeds.arstechnica.com/arstechnica/technology-lab",
        category="tech",
    ),
    RSSFeed(
        name="TechCrunch",
        url="https://techcrunch.com/feed/",
        category="tech",
    ),
    RSSFeed(
        name="The Verge",
        url="https://www.theverge.com/rss/index.xml",
        category="tech",
    ),
    RSSFeed(
        name="O'Reilly Radar",
        url="https://feeds.feedburner.com/oreilly/radar",
        category="tech",
    ),
    RSSFeed(
        name="Wired",
        url="https://www.wired.com/feed/rss",
        category="tech",
    ),
    RSSFeed(
        name="Engadget",
        url="https://www.engadget.com/rss.xml",
        category="tech",
    ),
    RSSFeed(
        name="Gizmodo",
        url="https://gizmodo.com/rss",
        category="tech",
    ),
    RSSFeed(
        name="VentureBeat",
        url="https://venturebeat.com/feed/",
        category="tech",
    ),
    # AI/ML News
    RSSFeed(
        name="AI News",
        url="https://www.artificialintelligence-news.com/feed/",
        category="ai",
    ),
    RSSFeed(
        name="OpenAI Blog",
        url="https://openai.com/blog/rss.xml",
        category="ai",
    ),
    RSSFeed(
        name="DeepMind Blog",
        url="https://deepmind.google/discover/blog/rss/",
        category="ai",
    ),
    RSSFeed(
        name="Hugging Face Blog",
        url="https://huggingface.co/blog/feed.xml",
        category="ai",
    ),
    # Developer News
    RSSFeed(
        name="GitHub Blog",
        url="https://github.blog/feed/",
        category="dev",
    ),
    RSSFeed(
        name="Stack Overflow Blog",
        url="https://stackoverflow.blog/feed/",
        category="dev",
    ),
    RSSFeed(
        name="Hacker News",
        url="https://news.ycombinator.com/rss",
        category="dev",
    ),
    RSSFeed(
        name="Reddit Programming",
        url="https://www.reddit.com/r/programming/.rss",
        category="dev",
    ),
    RSSFeed(
        name="Reddit MachineLearning",
        url="https://www.reddit.com/r/MachineLearning/.rss",
        category="ai",
    ),
]

# Feed URLs for quick access
FEED_URLS: list[str] = [feed.url for feed in DEFAULT_RSS_FEEDS]

# Category colors for UI
CATEGORY_COLORS: dict[str, str] = {
    "tech": "#3B82F6",  # Blue
    "ai": "#8B5CF6",    # Purple
    "dev": "#10B981",   # Green
}