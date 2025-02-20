from transformers import pipeline

# 加载模型和分词器（从你保存的目录加载）
ner = pipeline("token-classification", model="./ner_model", tokenizer="./ner_model", aggregation_strategy="simple")

# 输入示例文本
text = "为维护中华人民共和国国家主权和权益，合理开发利用海洋，有秩序地铺设和保护海底电缆、管道。"
result = ner(text)
print(result)
