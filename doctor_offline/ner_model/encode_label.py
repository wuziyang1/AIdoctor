#todo:04-数据处理阶段---把样本中标签转成数字然后保成DatasetDict类型数据
import pandas as pd
from datasets import Dataset, DatasetDict


def encode_label():


    label_to_index = {"O": 0, "B-dis": 1, "I-dis": 2, "B-sym": 3, "I-sym": 4}#把label转成数字

    # todo:将 csv 数据转换成 Dataset 类型
    train_data = pd.read_csv('ner_data/train.csv')
    valid_data = pd.read_csv('ner_data/valid.csv')

    train_data = Dataset.from_pandas(train_data)
    valid_data = Dataset.from_pandas(valid_data)

    corpus_data = DatasetDict({'train': train_data, 'valid': valid_data})
    '''
    corpus_data = DatasetDict({
        'train': train_data,
       'valid': valid_data
    })
    '''

    # todo:X不用动,将标签数据转换为索引表示，
    def data_handler(data_labels, data_inputs):

        data_label_ids = []#存储整个数据集转换后的结果
        for labels in data_labels:#data_labels是整个数据集
            label_ids = []#存储一句话转换后的结果
            for label in labels.split():#label是一句话的一个词对应的tag
                label_ids.append(label_to_index[label])
            data_label_ids.append(label_ids)

        return {'data_labels': data_label_ids, 'data_inputs': data_inputs}

    corpus_data = corpus_data.map(data_handler,input_columns=['data_labels', 'data_inputs'], batched=True)

    # 数据存储
    corpus_data.save_to_disk('ner_data/bilstm_crf_data_aidoc')

if __name__ == '__main__':
    encode_label()