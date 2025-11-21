#todo:工具函数---word2id以及padding segment

# todo:1-加载bert预训练模型
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#todo:直接使用bert模型中的tokenizer对文本进行张量化操作，不使用自己写的embedding层，因为数据量太少

# 你的 BERT 预训练模型所在的本地路径
model_path = r"E:\WorkSpace\Artificial_Intelligence\AIdoctor\doctor_offline\review_model\model\bert-base-chinese"

# 只加载 tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)
#如果使用bert模型就需要使用他的tokenizer

# 如果需要加载 BERT 模型
model = BertModel.from_pretrained(model_path).to(device)


'''
# 从本地加载
# source = '/root/.cache/torch/hub/huggingface_pytorch-transformers_main'
# 从github加载
source = 'huggingface/pytorch-transformers'

# 直接使用预训练的bert中文模型
model_name = 'bert-base-chinese'

# 通过torch.hub获得已经训练好的bert-base-chinese模型
# model =  torch.hub.load(source, 'model', model_name, source='local')
# 从github加载
model = torch.hub.load(source, 'model', model_name, source='github')

# 获得对应的字符映射器, 它将把中文的每个字映射成一个数字
# tokenizer = torch.hub.load(source, 'tokenizer', model_name, source='local')
# 从github加载
tokenizer = torch.hub.load(source, 'tokenizer', model_name, source='github')


'''

def get_bert_encode(text_1, text_2, mark=102, max_len=10):
    """
    对句子进行编码
    :param text_1: 第一个文本
    :param text_2: 第二个文本
    :param mark: 分隔符，一句话的结束符就是102
    :param max_len: 句子的最大长度
    :return:
    """
    #是同tokenizer对两个句子进行编码
    indexed_token = tokenizer.encode(text_1, text_2) # 得到的是每个字的编号 我:55

    # 先找到分隔标记的索引位置
    k = indexed_token.index(mark)

    # todo:处理第一句话：K前面的
    if len(indexed_token[:k]) >= max_len:
        indexed_token_1 = indexed_token[:max_len]
    else:
        indexed_token_1 = indexed_token[:k] + (max_len-len(indexed_token[:k]))*[0]

    # todo:处理第二句话：K后面的
    if len(indexed_token[k:]) >= max_len:
        indexed_token_2 = indexed_token[k:k+max_len]
    else:
        indexed_token_2 = indexed_token[k:] + (max_len-len(indexed_token[k:]))*[0]

    # todo:把两个处理后的句子拼接成一个
    indexed_token = indexed_token_1 + indexed_token_2

    segments_ids = [0]*max_len + [1]*max_len#相当于是起到标识作用的掩码

    segments_tensor = torch.tensor([segments_ids]).to(device)
    tokens_tensor = torch.tensor([indexed_token]).to(device)

    # 对每个字的编号进行编码操作，得到每个字的字向量，向量长度768
    with torch.no_grad():
        encoded_layers = model(tokens_tensor, token_type_ids=segments_tensor)[0]#[0]指的是取出最后一个隐藏层输出

    return encoded_layers

if __name__ == '__main__':
    text_1 = "人生该如何起头"
    text_2 = "改变要如何起手"
    encoded_layers = get_bert_encode(text_1, text_2)[0]
    print(encoded_layers)
    print(encoded_layers.shape)












