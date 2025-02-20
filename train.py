import json
from datasets import load_dataset, DatasetDict
from transformers import BertTokenizerFast, BertForTokenClassification, TrainingArguments, Trainer
import random

# 1. 加载数据（假设你的数据文件名为 "data.jsonl"）
data_files = {"data": "data.jsonl"}
raw_dataset = load_dataset("json", data_files=data_files)  # 默认读取每行一个JSON对象

# 2. 构建 BIO 标签集合
# 遍历整个数据集，提取所有出现的实体 label
label_set = set()
for ex in raw_dataset["data"]:
    for ent in ex["entities"]:
        label_set.add(ent["label"])
label_list = list(label_set)
label_list.sort()

# 构建 BIO 标签集合：对于每个实体 label，创建 "B-<label>" 和 "I-<label>"，再加上 "O"
bio_labels = ["O"] + [f"B-{l}" for l in label_list] + [f"I-{l}" for l in label_list]
# 构建映射
label2id = {label: idx for idx, label in enumerate(bio_labels)}
id2label = {idx: label for label, idx in label2id.items()}

print("BIO label set:", bio_labels)

# 3. 初始化分词器（使用本地 BERT 模型目录）
model_path = "D:/bert/bert"
tokenizer = BertTokenizerFast.from_pretrained(model_path)

# 4. 定义函数，将原始例子转换为模型输入，并根据字符偏移对齐标签
def tokenize_and_align_labels(example):
    text = example["text"]
    # 获取该例子中的所有实体，构造成 (start, end, label) 列表
    entity_spans = [(ent["start_offset"], ent["end_offset"], ent["label"]) for ent in example["entities"]]
    
    # 使用 tokenizer 对文本进行编码，并返回 offset_mapping
    tokenized_inputs = tokenizer(text, return_offsets_mapping=True, truncation=True, padding="max_length", max_length=128)
    offsets = tokenized_inputs.pop("offset_mapping")  # 每个 token 的 (start, end) 字符位置
    
    labels = []
    for start, end in offsets:
        # 对于特殊 token（例如 [CLS]、[SEP]），offset 通常为 (0,0)
        if start == end:
            labels.append(-100)
            continue
        token_label = "O"  # 默认
        # 遍历所有实体，判断该 token 是否落入实体范围内
        for ent_start, ent_end, ent_label in entity_spans:
            if start >= ent_start and end <= ent_end:
                # 如果 token 开始位置刚好等于实体开始位置，则为 B-标签；否则为 I-标签
                token_label = f"B-{ent_label}" if start == ent_start else f"I-{ent_label}"
                break
        labels.append(label2id[token_label])
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# 对数据集进行 map 处理（batched 模式下每个 example 都是一个字典列表）
tokenized_dataset = raw_dataset["data"].map(tokenize_and_align_labels, batched=False)

# 5. 自动划分数据集
# 先将整个数据集划分为 90% (train+val) 和 10% test
split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
# 从剩下的 90% 中再划分 10% 用作验证集，即验证集占整体数据大约 9%
train_val_split = split_dataset["train"].train_test_split(test_size=0.1111, seed=42)
dataset_dict = DatasetDict({
    "train": train_val_split["train"],
    "validation": train_val_split["test"],
    "test": split_dataset["test"]
})
print("数据集划分：")
print(dataset_dict)

# 6. 加载模型
num_labels = len(bio_labels)
model = BertForTokenClassification.from_pretrained(
    model_path,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    from_tf=True  # 告诉库这是TF模型权重，需要转换为PyTorch格式
)

# 7. 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_steps=20,
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir="./logs",
    seed=42
)

# 8. 创建 Trainer 对象
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["validation"]
)

# 9. 开始训练
trainer.train()

# 10. 训练完成后保存模型和分词器
model.save_pretrained("./ner_model")
tokenizer.save_pretrained("./ner_model")

# 11. 在测试集上评估模型
test_results = trainer.evaluate(dataset_dict["test"])
print("Test results:", test_results)
