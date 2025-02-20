BERT 实体识别项目
本项目基于 Hugging Face Transformers 库，通过微调 BERT 模型完成中文命名实体识别任务。项目包含以下内容：

数据预处理：从 JSONL 格式数据加载原始数据，并根据字符偏移对齐生成 BIO 格式的标签。
数据集划分：自动将数据集划分为训练集、验证集和测试集（例如 81% / 9% / 10%）。
模型训练：利用预训练的 TensorFlow 权重（转换为 PyTorch 格式）初始化 BertForTokenClassification 模型，并在自定义数据上微调。
模型评估：在验证集和测试集上评估模型效果。
模型调用示例：提供了使用 Hugging Face 的 pipeline 快速测试模型效果的代码示例，以及如何通过 Flask 部署 API 接口，供前端调用。

目录
环境要求
数据格式
安装与依赖
数据预处理与划分
模型训练
模型评估
模型调用示例
快速测试
后端 API 部署
项目结构

环境要求
Python 3.8+
PyTorch
Transformers
Datasets
TensorFlow（仅用于加载 TF 权重，请确保安装版本与需求一致）
Flask（如需要部署 API，可选）

数据格式
输入数据为 JSONL 格式，每行一个 JSON 对象，示例如下：
{"id":25316,"text":"为维护中华人民共和国国家主权和权益，合理开发利用海洋，有秩序地铺设和保护海底电缆、管道，制定本规定。","entities":[{"id":62,"label":"生产设施","start_offset":36,"end_offset":40},{"id":63,"label":"生产设施","start_offset":41,"end_offset":43},{"id":74,"label":"权利","start_offset":10,"end_offset":14}],"relations":[],"Comments":[]}
其中：

text：原始文本。
entities：实体列表，每个实体包含：
label：实体类别（如“生产设施”、“权利”等）。
start_offset 和 end_offset：实体在文本中的字符起始和结束位置。
本项目会基于这些信息生成 BIO 格式标签，并利用 BertTokenizerFast 的 offset_mapping 对齐 token 与标签。

安装与依赖
建议使用 conda 创建虚拟环境，然后安装依赖：
conda create -n myenv python=3.8
conda activate myenv
pip install transformers datasets flask
pip install tensorflow -i https://mirrors.aliyun.com/pypi/simple/
注意：由于加载 TensorFlow 权重，TensorFlow 必须安装。你也可以使用国内镜像加速安装。

数据预处理与划分
项目中的 train.py 脚本包含以下步骤：

加载数据：利用 load_dataset("json", data_files="data.jsonl") 加载 JSONL 数据。
构造 BIO 标签：遍历数据提取所有实体类别，生成 "B-<label>" 和 "I-<label>" 标签，构造标签映射。
对齐标签：使用 tokenizer 的 offset_mapping，将文本分词后与实体字符偏移对齐，生成每个 token 的标签（特殊 token 设置为 -100）。
数据集拆分：将整个数据集自动划分为训练集、验证集和测试集。

模型训练
项目使用 Hugging Face 的 BertForTokenClassification 加载预训练 TensorFlow 权重（通过 from_tf=True 参数转换为 PyTorch 权重），并在数据上进行微调。训练日志显示 loss 从较高值逐步下降，验证集和测试集上评估指标正常。

你可以运行以下命令开始训练（确保修改代码中的文件路径等参数）：
python train.py
训练完成后，模型和分词器会保存在 ./ner_model 目录下。

模型评估
训练过程中会在每个 epoch 后在验证集上评估模型，训练结束后你还可以在测试集上运行评估，查看最终的 eval_loss 等指标。评估结果会打印在日志中。

模型调用示例
快速测试
你可以使用 Hugging Face 的 pipeline 快速加载和调用模型进行实体识别测试。创建一个简单的 Python 脚本，例如 test.py：
from transformers import pipeline

# 从保存的模型目录加载模型和分词器
ner = pipeline("token-classification", model="./ner_model", tokenizer="./ner_model", aggregation_strategy="simple")

# 输入示例文本
text = "为维护中华人民共和国国家主权和权益，合理开发利用海洋，有秩序地铺设和保护海底电缆、管道。"
result = ner(text)
print(result)
运行后，会输出模型识别的实体及其类别、分数和位置范围。

后端 API 部署
如果需要将模型集成到后端服务，可以使用 Flask 构建一个简单的 API。以下是一个示例代码（保存为 app.py）：
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
启动该 Flask 服务后，当前端通过 POST 请求发送带有 "text" 字段的 JSON 数据时，后端会调用 BERT 实体识别模型，并将识别结果返回给前端。

项目结构
├── data.jsonl          # 原始数据文件（JSONL 格式）
├── train.py            # 数据加载、预处理、训练及评估代码
├── test_model.py       # 快速测试模型调用示例
├── app.py              # Flask API 部署示例
├── ner_model/          # 训练完成后保存的模型和分词器目录
├── README.md           # 本文件
└── requirements.txt    # 项目依赖（可选）
