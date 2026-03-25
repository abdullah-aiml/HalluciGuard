from transformers import AutoModelForSequenceClassification

model_name = "cross-encoder/nli-deberta-v3-base"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
print("Labels:", model.config.id2label)
