from voxmind.services.nlp.intent_classifier import classify_intent


def test_greeting():
    assert classify_intent("Hello there!") == ("greeting", 1.0)


def test_gratitude():
    label, confidence = classify_intent("Thanks a lot for your help")
    assert label == "gratitude"
    assert confidence == 1.0


def test_wh_question():
    label, _ = classify_intent("What time is the meeting?")
    assert label == "question"


def test_yes_no_question_without_question_mark_prefix_still_detected_via_mark():
    label, _ = classify_intent("is this the right document?")
    assert label == "question"


def test_command():
    label, _ = classify_intent("Please summarize this document")
    assert label == "command"


def test_statement_fallback_has_lower_confidence():
    label, confidence = classify_intent("The quarterly report is attached.")
    assert label == "statement"
    assert confidence == 0.5
