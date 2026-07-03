import streamlit as st

from input_prep import load_training_stats, prepare_model_input
from predict import predict_income
from llm_parser import parse_user_text


stats = load_training_stats("data/adult.csv")


def main():
    st.title("Adult Income Prediction Assistant")

    st.write(
        "Describe a person's demographic and work information in natural language, "
        "and this app will estimate whether their income is more likely to be above or below 50K."
    )

    user_text = st.text_area(
        "Enter the description here:",
        height=180,
        placeholder=(
            "Example: I am a 45-year-old married male, working full-time in private industry "
            "as an executive manager. I have a masters degree, work 50 hours per week, "
            "and live in the United States."
        ),
    )

    if st.button("Predict income class"):
        if not user_text.strip():
            st.warning("Please enter a description first.")
            return

        parsed = parse_user_text(user_text)

        if parsed["out_of_scope"]:
            st.error("This question is outside the scope of income prediction based on the Adult dataset.")
            return

        if parsed["clarification_needed"]:
            st.warning(parsed["clarification_question"])
            st.write("Extracted features so far:")
            st.json(parsed["features"])
            return

        features = parsed["features"]

        model_input = prepare_model_input(features, stats)
        result = predict_income(model_input)

        prediction = result["prediction"]
        probs = result["probabilities"]

        prob_low = probs[0]
        prob_high = probs[1]

        st.subheader("Prediction result")
        st.write(f"Predicted income class: **{prediction}**")
        st.write(f"Estimated probability of income >50K: **{prob_high:.2%}**")
        st.write(f"Estimated probability of income <=50K: **{prob_low:.2%}**")

        st.subheader("Parsed features")
        st.json(features)

        st.caption(
            "Note: This prediction is based on the Adult Income dataset and a trained ML model. "
            "It is a statistical estimate, not a guarantee."
        )


if __name__ == "__main__":
    main() 

    

