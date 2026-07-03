import json
import re
from typing import Any, Dict


def parse_user_text(user_text: str) -> Dict[str, Any]:
    text = user_text.lower()

    age_match = re.search(r"\b(\d{2})[- ]?year[- ]old\b", text)
    if not age_match:
        age_match = re.search(r"\bage\s+(\d{2})\b", text)
    age = int(age_match.group(1)) if age_match else None

    sex = None
    if " male" in text or " man" in text or " husband" in text:
        sex = "Male"
    elif " female" in text or " woman" in text or " wife" in text:
        sex = "Female"

    workclass = None
    if " government" in text or "gov" in text:
        workclass = "State-gov"
    elif " self-employed" in text or " self employed" in text:
        workclass = "Self-emp-not-inc"
    elif " private" in text or "company" in text or "industry" in text:
        workclass = "Private"

    education = None
    if "high school" in text or "hs diploma" in text:
        education = "HS-grad"
    elif "bachelor" in text or "ba " in text or "bs " in text:
        education = "Bachelors"
    elif "master" in text or "masters" in text:
        education = "Masters"
    elif "doctorate" in text or "phd" in text:
        education = "Doctorate"

    marital_status = None
    if "married" in text or "spouse" in text:
        marital_status = "Married-civ-spouse"
    elif "divorced" in text:
        marital_status = "Divorced"
    elif "never married" in text or "single" in text:
        marital_status = "Never-married"

    occupation = None
    if "manager" in text or "executive" in text:
        occupation = "Exec-managerial"
    elif "tech" in text or "engineer" in text:
        occupation = "Tech-support"
    elif "sales" in text:
        occupation = "Sales"
    elif "teacher" in text:
        occupation = "Prof-specialty"
    elif "clerical" in text or "admin" in text:
        occupation = "Adm-clerical"

    relationship = None
    if "husband" in text:
        relationship = "Husband"
    elif "wife" in text:
        relationship = "Wife"
    elif "unmarried" in text or "single" in text:
        relationship = "Unmarried"
    elif marital_status == "Married-civ-spouse" and sex == "Male":
        relationship = "Husband"
    elif marital_status == "Married-civ-spouse" and sex == "Female":
        relationship = "Wife"

    race = None

    capital_gain = 0
    capital_loss = 0

    hours_match = re.search(r"(\d+)\s+hours", text)
    hours_per_week = int(hours_match.group(1)) if hours_match else None

    native_country = None
    if " united states" in text or " u.s." in text or " usa" in text:
        native_country = "United-States"

    features = {
        "age": age,
        "workclass": workclass,
        "education": education,
        "marital-status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours_per_week,
        "native-country": native_country,
    }

    required = [
        "age",
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "sex",
        "hours-per-week",
        "native-country",
    ]
    missing_fields = [f for f in required if features[f] is None]

    clarification_needed = len(missing_fields) > 0
    clarification_question = ""
    if clarification_needed:
        clarification_question = (
            "Please provide the following missing information: "
            + ", ".join(missing_fields)
            + "."
        )

    result = {
        "features": features,
        "missing_fields": missing_fields,
        "out_of_scope": False,
        "clarification_needed": clarification_needed,
        "clarification_question": clarification_question,
    }

    return result


if __name__ == "__main__":
    sample_text = (
        "I am a 45-year-old married male working in private industry as an executive manager. "
        "I have a masters degree, I work 50 hours per week, and I live in the United States."
    )
    result = parse_user_text(sample_text)
    print(json.dumps(result, indent=2))

    



 

    







