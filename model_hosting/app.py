import streamlit as st
import dill as pickle
from utils_models import load_roberta, predict_roberta, predict_lr_rf
from text_preprocessing import preprocess_text

def identity_tokenizer(text):
  return text

# Load the saved vectorizer, logistics regression, random forest, roberta models
with open('model_hosting/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('model_hosting/lr_classifier_default.pkl', 'rb') as f:
    lr_model = pickle.load(f)

with open('model_hosting/rf_classifier_param.pkl', 'rb') as f:
    rf_model = pickle.load(f)

hf_model, tokenizer = load_roberta()

# Streamlit app
def main():
    st.title('Text Classification App')

    # Text input box
    text_input = st.text_area("Enter your text here:", "")

    # Preprocess text before passing to models
    text_mod, lemmatized = preprocess_text(text_input)
    
    # Button to trigger predictions
    if st.button("Predict"):
        # Predictions from RoBERTa model
        roberta_pred = predict_roberta(hf_model, tokenizer, text_mod)
        # Predictions from Logistic Regression and Random Forest models
        lr_prediction = predict_lr_rf(lr_model, tfidf_vectorizer, lemmatized)

        rf_prediction = predict_lr_rf(rf_model, tfidf_vectorizer, lemmatized)

        # Display predictions
        st.write("RoBERTa Prediction:", roberta_pred)
        st.write("Logistic Regression Prediction:", lr_prediction)
        st.write("Random Forest Prediction:", rf_prediction)

if __name__ == "__main__":
    main()
