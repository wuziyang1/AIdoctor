import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel

# 初始化BERT组件
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert_model = BertModel.from_pretrained('bert-base-chinese')

# 配置参数
HIDDEN_SIZE = 128
OUTPUT_SIZE = 2  # 二分类
BATCH_SIZE = 2  # 因BERT较大，适当减小batch size
NUM_EPOCHS = 10
LR = 2e-5  # 更小的学习率
MAX_LEN = 32  # BERT最大序列长度


# 修改后的BERT编码函数（支持批量）
def get_bert_encode(texts):
    """
    批量处理BERT编码
    :param texts: 文本列表
    :return: 编码后的张量 (batch_size, seq_len, 768)
    """
    inputs = tokenizer(
        texts,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = bert_model(**inputs)

    return outputs.last_hidden_state


# 数据预处理类
class BertDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # 训练时动态编码（生产环境建议预编码）
        encoded = get_bert_encode([self.texts[idx]])[0]  # (1, seq_len, 768) -> (seq_len, 768)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return encoded, label


# 示例数据
raw_data = [
    (1, "手掌软硬度异常"),
    (0, "常异度硬软掌手"),
    (1, "多发性针尖样瘀点"),
    (0, "点瘀样尖针性发多"),
    (1, "肩臂外侧的感觉障碍和功能受限"),
    (0, "限受能功和碍障觉感的侧外臂肩"),
    (1, "下肢或周身软瘫"),
    (0, "瘫软身周或肢下"),
    (1, "累及整个腹腔的腹膜炎")
]

texts = [item[1] for item in raw_data]
labels = [item[0] for item in raw_data]

# 拆分数据集
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42)

# 创建数据加载器
train_dataset = BertDataset(train_texts, train_labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = BertDataset(val_texts, val_labels)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# 修改后的RNN模型
class BERTRNN(nn.Module):
    def __init__(self, bert_model, hidden_size, output_size):
        super(BERTRNN, self).__init__()
        self.bert = bert_model
        self.rnn = nn.RNN(
            input_size=768,  # BERT输出维度
            hidden_size=hidden_size,
            batch_first=True
        )
        self.classifier = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # BERT编码
        with torch.no_grad():  # 固定BERT参数
            outputs = self.bert(input_ids=x['input_ids'],
                                attention_mask=x['attention_mask'])

        # RNN处理
        rnn_out, _ = self.rnn(outputs.last_hidden_state)

        # 取最后一个时间步
        last_hidden = rnn_out[:, -1, :]

        # 分类
        return self.classifier(self.dropout(last_hidden))


# 初始化模型
model = BERTRNN(bert_model, HIDDEN_SIZE, OUTPUT_SIZE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


# 训练函数（需修改前处理）
def train():
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        for inputs, labels in train_loader:
            # 重组输入格式
            batch_encoding = tokenizer(
                [t for t in inputs],  # 这里需要实际文本输入
                max_length=MAX_LEN,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )

            # 前向传播
            outputs = model(batch_encoding)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 验证流程
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                batch_encoding = tokenizer(
                    [t for t in inputs],
                    max_length=MAX_LEN,
                    padding='max_length',
                    truncation=True,
                    return_tensors="pt"
                )
                outputs = model(batch_encoding)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"Train Loss: {total_loss / len(train_loader):.4f}")
        print(f"Val Acc: {100 * correct / total:.2f}%\n")


# 开始训练
train()