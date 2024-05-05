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

text_area_count = 0  # Counter for text area widgets
    
while True:
    text_area_count += 1
    # Text input box with unique key
    text_input = st.text_area(f"Enter your text {text_area_count}:", key=f"text_area_{text_area_count}")

    # Preprocess text before passing to models
    text_mod, lemmatized = preprocess_text(text_input)
    
    # Button to trigger predictions
    if st.button("Класифікувати"):
        # Predictions from RoBERTa model
        roberta_pred = predict_roberta(hf_model, tokenizer, text_mod)
        # Predictions from Logistic Regression and Random Forest models
        lr_prediction = predict_lr_rf(lr_model, tfidf_vectorizer, lemmatized)
        rf_prediction = predict_lr_rf(rf_model, tfidf_vectorizer, lemmatized)

        # Display predictions
        st.write("RoBERTa Prediction:", roberta_pred)
        st.write("Logistic Regression Prediction:", lr_prediction)
        st.write("Random Forest Prediction:", rf_prediction)

    # Button to add another text input
    if st.button("Завершити"):
        break

if __name__ == "__main__":
    main()
