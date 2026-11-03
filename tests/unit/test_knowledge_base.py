def test_return_query_retrieves_30_day_policy(kb):
    context = kb.retrieve("What is your return policy?")
    assert any("30 days" in passage for passage in context)


def test_unrelated_query_returns_no_context(kb):
    assert kb.retrieve("Do you sell cars on Mars?") == []


def test_get_returns_exact_policy_text(kb):
    assert "30 days" in kb.get("returns")