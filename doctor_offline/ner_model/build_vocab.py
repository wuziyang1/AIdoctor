#todo:02-词表预处理
import json
from tqdm import tqdm

def build_vocab():
    """
    处理json文件，读取key，并存入txt文件
    只用到了词表的key
    :return:
    """
    chat_to_id = json.load(open('ner_data/char_to_id.json', mode='r', encoding='utf-8'))
    unique_words = list(chat_to_id.keys())[1:-1]#去掉一头一尾
    unique_words.insert(0, '[UNK]')
    unique_words.insert(0, '[PAD]')

    # 把数据写入到文本中
    with open('ner_data/bilstm_crf_vocab_aidoc.txt', 'w') as file:
        for word in unique_words:
            file.write(word+'\n')

#todo:根据训练集构建自己的词表
def build_vocab(file_path, tokenizer, max_size, min_freq):

    UNK, PAD, CLS = "[UNK]", "[PAD]", "[CLS]"
    """
    构建词汇表的函数。

    参数：
    - file_path (str): 包含文本数据的文件路径。就是训练集的18万条数据
    - tokenizer (function): 用于分词的函数，接受一个字符串并返回分词后的结果。
    - max_size (int): 词汇表的最大大小，即保留的词汇数量上限。
    - min_freq (int): 词汇表中词语的最小出现频率，低于此频率的词汇将被过滤掉。
    """
    vocab_dic = {}  # 用于存储词汇表的字典，键为单词，值为单词出现的次数
    with open(file_path, "r", encoding="UTF-8") as f:
        for line in tqdm(f):#tqdm进度条
            line = line.strip()
            if not line:
                continue
            content = line.split("\t")[0]  # 以制表符分隔的文本，这里取第一列的内容
            # 使用给定的分词器（tokenizer）对文本进行分词，并更新词汇表
            for word in tokenizer(content):#tokenizer相当于jieba
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        # 根据词频对词汇表进行排序，并选择出现频率较高的词汇
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq],
                            key=lambda x: x[1], reverse=True)[:max_size]
        # 将选定的词汇构建为字典，键为单词，值为索引
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        # 添加特殊符号到词汇表，例如未知符号（UNK）、填充符号（PAD）
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic




if __name__ == '__main__':
    # build_vocab()


    tokenizer = lambda x: [y for y in x]
    vocab = build_vocab("./ner_data/train.txt", tokenizer=tokenizer, max_size=10000, min_freq=1)
    print(len(vocab))
    for i, (key, value) in enumerate(vocab.items()):
        if i >= 200:
            break
        print(f"{key}: {value}")
