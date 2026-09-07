from voxmind.services.nlp.topic_extractor import extract_topics


def test_extracts_most_frequent_non_stopwords_in_frequency_order():
    text = "The database migration failed. The migration failed again on the database server."
    topics = extract_topics(text, max_topics=3)
    assert topics[0] in ("migration", "database", "failed")
    assert "the" not in topics
    assert len(topics) <= 3


def test_empty_text_yields_no_topics():
    assert extract_topics("") == []
    assert extract_topics("the a an") == []


def test_deterministic_across_repeated_calls():
    text = "Speaker emotion analysis requires acoustic features and speaker embeddings."
    assert extract_topics(text) == extract_topics(text)
