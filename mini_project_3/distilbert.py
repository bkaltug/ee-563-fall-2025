import time
from transformers import pipeline

distilbert = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

text = "I've loved HuggingFace courses my whole life."

init = time.time()
score = distilbert(text)
end = time.time()
process_time = end - init

print(f"Distilbert positivity score: {score}\n Process time: {process_time:.5f} seconds\n")
