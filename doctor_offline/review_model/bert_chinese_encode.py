import torch
from transformers import BertModel, BertTokenizer

#todo:直接使用bert模型中的tokenizer对文本进行张量化操作，不使用自己写的embedding层，因为数据量太少

# 你的 BERT 预训练模型所在的本地路径
# model_path = "/export/data/AIdoctor/doctor_offline/review_model/model/bert-base-chinese"
model_path = r"E:\WorkSpace\Artificial_Intelligence\AIdoctor\doctor_offline\review_model\model\bert-base-chinese"#windows下训练

# 只加载 tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)

# 如果需要加载 BERT 模型
model = BertModel.from_pretrained(model_path)



'''
# 从本地加载
source = './model/bert-base-chinese'
# 直接使用预训练的bert中文模型
model_name = 'bert-base-chinese'
# 通过torch.hub获得已经训练好的bert-base-chinese模型
model =  torch.hub.load(source, 'model', model_name, source='local')
# 获得对应的字符映射器, 它将把中文的每个字映射成一个数字
tokenizer = torch.hub.load(source, 'tokenizer', model_name, source='local')
#tokenizer记录的是每个字的编码信息，相当于一个字典。一个字对应一个编码：比如：‘你’对应245

'''

'''
# 从github加载
model_name = "bert-base-chinese"
source = "huggingface/pytorch-transformers"
model = torch.hub.load(source, 'model', model_name, source='github')
tokenizer = torch.hub.load(source, 'tokenizer', model_name, source='github')
'''


def get_bert_encode_for_single(text):
    """
    使用bert-base-chinese模型对文本进行编码
    :param text:  输入的文本
    :return: 编码后的张量
    """
    # 通过tokenizer对文本进行编号
    indexed_tokens = tokenizer.encode(text)[1: -1]#[1: -1]把101和102切掉
    # print(indexed_tokens)
    # 把列表转成张量
    tokens_tensor = torch.LongTensor([indexed_tokens])

    # 不自动进行梯度计算
    with torch.no_grad():
        output = model(tokens_tensor)

    # print(output)
    return output[0]

if __name__ == '__main__':
    text = "你好，福建舰"
    outputs = get_bert_encode_for_single(text)
    print('text编码:', outputs)

'''
torch 1.1
transformers4.17
如果数据量较少，不适合训练带有enbedding层的模型
'''