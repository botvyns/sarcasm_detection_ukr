import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def map_num_to_label(num):
    return 'сарказм' if num==1 else 'не сарказм'

def load_roberta(): 
    model_ckpt = "Snizhanna/ukr-roberta-base-finetuned-sarc"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    id2label = {1: "sarcastic",0: "not_sarcastic"}
    label2id = {"sarcastic": 1, "not_sarcastic": 0}
    hf_model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=2, label2id=label2id, id2label=id2label)
    return hf_model, tokenizer

def predict_roberta(model, tokenizer, text):
    tokenized_input = tokenizer(text, return_tensors="pt")
    predictions = model(**tokenized_input)
    prediction = predictions.logits.argmax().item()
    return map_num_to_label(prediction)

def predict_lr_rf(model, vectorizer, text):
    prediction = model.predict(vectorizer.transform([text]))
    return map_num_to_label(prediction)
