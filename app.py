from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# 加载模型（启动时加载，避免每次请求重复加载）
ner_pipeline = pipeline(
    "token-classification",
    model="./ner_model",
    tokenizer="./ner_model",
    aggregation_strategy="simple"
)

@app.route("/ner", methods=["POST"])
def ner_endpoint():
    input_text = request.json.get("text", "")
    if not input_text:
        return jsonify({"error": "No text provided"}), 400
    result = ner_pipeline(input_text)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
