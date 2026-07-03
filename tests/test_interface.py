from src.llm_parser import parse_user_text


def test_parse_user_text_extracts_expected_features():
    user_text = (
        "I am a 45-year-old married male working in private industry as an executive manager. "
        "I have a masters degree, I work 50 hours per week, and I live in the United States."
    )

    result = parse_user_text(user_text)
    features = result["features"]

    assert features["age"] == 45
    assert features["workclass"] == "Private"
    assert features["education"] == "Masters"
    assert features["marital-status"] == "Married-civ-spouse"
    assert features["occupation"] == "Exec-managerial"
    assert features["relationship"] == "Husband"
    assert features["sex"] == "Male"
    assert features["hours-per-week"] == 50
    assert features["native-country"] == "United-States"
    assert result["clarification_needed"] is False


def test_parse_user_text_handles_missing_fields_gracefully():
    user_text = "I am a 30-year-old man."

    result = parse_user_text(user_text)

    assert result["clarification_needed"] is True
    assert "workclass" in result["missing_fields"]
    assert "education" in result["missing_fields"]
    assert "hours-per-week" in result["missing_fields"]
    assert result["clarification_question"] != ""

    