from app.services.web_search import WebSearchService


def test_build_fallback_query_drops_year_ranges_and_trims_length():
    service = WebSearchService()

    refined = service._build_fallback_query(
        "OpenAI Azure military access timeline 2023-2025 highest classification approval"
    )

    assert "2023-2025" not in refined
    assert refined == "OpenAI Azure military access timeline highest classification"


def test_build_fallback_query_keeps_short_query():
    service = WebSearchService()

    refined = service._build_fallback_query("Azure OpenAI 最高密级授权 美国国防部 2025")

    assert refined == "Azure OpenAI 最高密级授权 美国国防部"
