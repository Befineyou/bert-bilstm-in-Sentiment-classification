# 确定编码工具
from transformers import BertTokenizer
# 读取数据集
import pandas as pd
from datasets import Dataset
# 定义数据集
import torch
from transformers import AdamW
from transformers.optimization import get_scheduler

# 确定编码工具
token = BertTokenizer.from_pretrained('bert-base-chinese')
print(token)

# 读取已保存的 CSV 文件
train_data = pd.read_csv('data/balanced_train_data.csv')
test_data = pd.read_csv('balanced_test_data.csv')
val_data = pd.read_csv('data/balanced_val_data.csv')

device = 'cpu'
if torch.cuda.is_available():
    device = 'CUDA'
print(device)


# 构建 Hugging Face 数据集

# 控制训练数据量
# train_subset_size = 1600  # 设置每次训练使用的数据量
# train_subset = train_data.iloc[:train_subset_size]
train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)
val_dataset = Dataset.from_pandas(val_data)
# 控制测试数据量
# test_subset_size = 320 # 设置每次测试使用的数据量
# test_subset = test_data.iloc[:test_subset_size]
# train_dataset = Dataset.from_pandas(train_data)
# test_dataset = Dataset.from_pandas(test_data)
# 保存Hugging Fae数据集
# train_dataset.save_to_disk('train_dataset')
# test_dataset.save_to_disk('test_dataset')
print(train_dataset)

#定义数据集
import torch

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        text = self.dataset[item]['text']
        label = self.dataset[item]['label']
        return text, label

# 使用示例
train_dataset = CustomDataset(train_dataset)
test_dataset = CustomDataset(test_dataset)
val_dataset = CustomDataset(val_dataset)

print(len(train_dataset), train_dataset[20])




# 定义数据整理函数
def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]
    data = token.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=500,
                                   return_tensors='pt',
                                   return_length=True)
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    #labels = torch.LongTensor(labels)
    # labels = torch.tensor(labels).long()
    labels = torch.tensor([label if label != -1 else 0 for label in labels]).long()
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    token_type_ids = token_type_ids.to(device)
    labels = labels.to(device)

    return input_ids, attention_mask, token_type_ids, labels

# 定义数据集加载器

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                     batch_size=16,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                     batch_size=16,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                     batch_size=16,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)

print(len(train_loader))

from transformers import BertModel

pretrained = BertModel.from_pretrained('bert-base-chinese')
for param in pretrained.parameters():
    param.requires_grad_(False)
# https://blog.csdn.net/hhhhhhhhhhwwwwwwwwww/article/details/121289547
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class BertBiLSTMClassifier(nn.Module):
    def __init__(self, num_classes, hidden_size=768, lstm_hidden_size=128, lstm_layers=1):
        super(BertBiLSTMClassifier, self).__init__()

        # # 加载预训练的BERT模型和分词器
        # self.bert = BertModel.from_pretrained(bert_model_name)
        # self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        # # 冻结BERT的参数
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        # BiLSTM层
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=lstm_hidden_size, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True)
        # 全连接层用于分类
        self.fc = nn.Linear(lstm_hidden_size * 2, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # BERT的前向传播
        with torch.no_grad():
            outputs = pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        #pooled_output = outputs.pooler_output
        last_hidden_state = outputs.last_hidden_state
        # 将BERT输出输入BiLSTM
        lstm_out, _ = self.lstm(last_hidden_state)
        #lstm_out, _ = self.lstm(pooled_output)

        #lstm_out, _ = self.lstm(pooled_output.unsqueeze(0))

        # 提取BiLSTM的最后一层输出
        lstm_out = lstm_out[:, -1, :]
        #lstm_out = lstm_out[:, -1, :, :]

        # 全连接层分类
        logits = self.fc(lstm_out)

        return logits


# 定义模型和输入
num_classes = 3  # 你的分类类别数量
model = BertBiLSTMClassifier(num_classes).to(device)

# 输出模型结构
print(model)


# def train():
#     optimizer = AdamW(model.parameters(), lr=5e-4)
#     criterion = torch.nn.CrossEntropyLoss()
#     scheduler = get_scheduler(name='linear',
#                               num_warmup_steps=0,
#                               num_training_steps=len(train_loader),
#                               optimizer=optimizer)
#     model.train()
#     for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
#         out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#         loss = criterion(out, labels)
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
#         optimizer.zero_grad()
#         if i % 10 == 0:
#             out = out.argmax(dim=1)
#             accuracy = (out == labels).sum().item() / len(labels)
#             lr = optimizer.state_dict()['param_groups'][0]['lr']
#             print(i, loss.item(), lr, accuracy)
#         if i % 100 == 0:
#             torch.save(model.state_dict(), f'bert_bilstm_model_epoch_{i}.pth')
# train()
optimizer = AdamW(model.parameters(), lr=5e-4)
criterion = torch.nn.CrossEntropyLoss()
scheduler = get_scheduler(name='linear',num_warmup_steps=0,num_training_steps=len(train_loader),optimizer=optimizer)

def train_and_validate(model, train_loader, val_loader, optimizer, criterion, scheduler, num_epochs=2):
    for epoch in range(num_epochs):
        model.train()
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
            out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if i % 10 == 0:
                out = out.argmax(dim=1)
                accuracy = (out == labels).sum().item() / len(labels)
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                print(f'Train Epoch {epoch}, Step {i}, Loss: {loss.item()}, LR: {lr}, Accuracy: {accuracy}')

            if i % 100 == 0:
                torch.save(model.state_dict(), f'bert_bilstm_model_epoch_{epoch}_step_{i}.pth')

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for val_input_ids, val_attention_mask, val_token_type_ids, val_labels in val_loader:
                val_out = model(input_ids=val_input_ids, attention_mask=val_attention_mask, token_type_ids=val_token_type_ids)
                val_loss += criterion(val_out, val_labels).item()

                val_out = val_out.argmax(dim=1)
                correct += (val_out == val_labels).sum().item()
                total += len(val_labels)

        val_accuracy = correct / total
        average_val_loss = val_loss / len(val_loader)
        print(f'Validation Epoch {epoch}, Loss: {average_val_loss}, Accuracy: {val_accuracy}')

        # You can add early stopping criteria here based on validation performance if needed.

train_and_validate(model,train_loader, val_loader, optimizer, criterion, scheduler)



# 保存模型
# torch.save(model.state_dict(), 'bert_bilstm_model.pth')
def test():
    model.eval()
    correct = 0
    total = 0
    for i,(input_ids,attention_mask,token_type_ids,labels) in enumerate(test_loader):
        if i==5:
            break
        print(i)
        with torch.no_grad():
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids = token_type_ids)
            #out = out.argmax(dim=1)
            out = torch.argmax(out, dim=1)
            correct +=(out==labels).sum().item()
            total +=len(labels)
        print(correct/total)
test()
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def test():
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(test_loader):
        if i == 5:
            break

        with torch.no_grad():
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)

            out = torch.argmax(out, dim=1)
            correct += (out == labels).sum().item()
            total += len(labels)

            # Collect true labels and predicted labels for later evaluation
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(out.cpu().numpy())

        print(f"Accuracy: {correct / total}")

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Final Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")


test()

# # 加载模型
# loaded_model = BertBiLSTMClassifier(num_classes)
# loaded_model.load_state_dict(torch.load('bert_bilstm_model.pth'))
# loaded_model.eval()  # 设置为评估模式
