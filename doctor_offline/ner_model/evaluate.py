#todo:06
import torch
from datasets import load_from_disk
from transformers import BertTokenizer
from bilstm_crf import NER

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#todo:1-
def evaluate(model=None, tokenizer=None, data=None):
    """
    评估模型
    :param model: 如果是None，表示评估用的模型是已经训练好的，否则就是训练过程中传入的
    """
    # todo:1 加载验证集，获取真实实体
    if data is None:
        data = load_from_disk('ner_data/bilstm_crf_data_aidoc')['train']

    # todo:2-统计不同类别实体的数量(获取真实值)
    total_entities = get_true_entitie(data) #{'DIS': [], 'SYM': []}

    # todo:3-使用模型对验证集进行验证(获取预测值)
    model_entities = get_pred_entities(data, model, tokenizer) #{'DIS': [], 'SYM': []}

    #todo:4-开始计算指标
    indicators = cal_prf(model_entities, total_entities)

    return indicators

#todo:4-开始计算指标
def cal_prf(model_entities, total_entities):#(真实值，预测值)
    # 3 指标计算
    # 每个类别实体 精确率 召回率
    # 整个模型在所有数据上准确率
    #total_entities：{'DIS': [], 'SYM': []}
    #model_entities：{'DIS': [], 'SYM': []}

    indicators = []  # 保存计算的指标

    total_pred_correct = 0#预测值
    total_true_correct = 0#真实值

    for key in ['DIS', 'SYM']:  # total_entities.keys(): # key的取值这里就两个 DIS SYM
        true_entities = total_entities[key]# DIS和SYM的
        true_entities_num = len(true_entities)  # TP+FN 真是样本DIS和SYM的个数

        pred_entities = model_entities[key]
        pred_entities_num = len(pred_entities)  # TP+FP 预测样本DIS和SYM的个数

        # 计算TP，判断预测正确的数量。
        pred_correct = 0  # TP:我预测A是正样本且A本来就是正样本
        pred_incorrect = 0  # FP:我预测是正样本但是他不是正样本
        for pred_entity in pred_entities:#循环取出预测结果中的每一个，看看他是否在真实结果中存在
            if pred_entity in true_entities:
                pred_correct += 1#预测正确
                continue
            pred_incorrect += 1#预测错误

        # 统计两轮循环中一共预测正确的数量和全部实体的数量
        total_pred_correct += pred_correct       #这个是模型预测对的正样本个数，也就是模型预测他为正样本，它本身也是正样本的个数
        total_true_correct += true_entities_num  #这个是总的正样本个数

        # recall precision f1
        recall = pred_correct / true_entities_num if true_entities_num!=0 else 0
        precision = pred_correct / pred_entities_num if pred_entities_num!=0 else 0

        f1 = 0
        if recall != 0 or precision != 0:
            f1 = 2 * recall * precision / (recall + precision)

        print(key, '查全率: %.3f' % recall)
        print(key, '查准率: %.3f' % precision)
        print(key, 'f1: %.3f' % f1)
        print('-' * 50)

        indicators.append([recall, precision, f1])


    print('准确率：%.3f' % (total_pred_correct / total_true_correct))
    indicators.append(total_pred_correct / total_true_correct)

    return indicators


def get_pred_entities(data, model, tokenizer):
    # 2 使用模型对验证集进行预测
    if model is None:
        model_param = torch.load('model/BiLSTM-CRF-300.bin',map_location=device)
        model = NER(**model_param['init'])
        model.load_state_dict(model_param['state'])
        model.to(device)

    #todo:构建分词器
    if tokenizer is None:
        tokenizer = BertTokenizer(vocab_file='ner_data/bilstm_crf_vocab_aidoc.txt')
    model_entities = {'DIS': [], 'SYM': []}#存的是模型预测值

    def start_evaluate(data_inputs):
        # 对输入的原始文本的每个字进行编号
        model_inputs = tokenizer.encode(data_inputs, return_tensors='pt', add_special_tokens=False)[0]
        model_inputs = model_inputs.to(device)

        # 使用模型进行预测
        with torch.no_grad():
            label_list = model.predict(model_inputs)

        # 通过extract_decode()进行实体提取并保存
        extract_entities = extract_decode(label_list, ''.join(data_inputs.split()))

        nonlocal model_entities

        for key, value in extract_entities.items():
            model_entities[key].extend(value)

    data.map(start_evaluate, input_columns=['data_inputs'], batched=False)

    return model_entities

# todo:2-统计不同类别实体的数量(获取真实实体)

def get_true_entitie(data):#data是整个验证集或者一个batch的训练集
    total_entities = {'DIS': [], 'SYM': []}

    def calculate_handler(data_inputs, data_labels):#这儿的数据十一条一条地传进来的
        # todo:去掉输入数据中的空格
        data_inputs = ''.join(data_inputs.split())

        #todo:提取句子中的实体-这里传进来的是真实标签，所以获取的是真实实体
        extract_entities = extract_decode(data_labels, data_inputs) #{'DIS':[], 'SYM': []}
        #这里把标签和数据都送进去得到的是真实值

        #todo:统计每种实体的数量
        nonlocal total_entities
        for key, value in extract_entities.items():
            total_entities[key].extend(value)

    #统计不同实体的数量
    data.map(calculate_handler, input_columns=['data_inputs', 'data_labels'])

    return total_entities


#todo:3-提取句子中的实体
def extract_decode(data_labels, data_inputs):#一句话的label和文本
    """
    根据输入的标签序列和文本，从文本中提取实体
    :param data_labels: 标签序列
    :param data_inputs: 文本
    一共两次调用这个函数，一次传进来的是真实标签一次传进来的是预测标签

    # data_inputs: 我     肚   子       疼  呀
    # data_labels:[0, B-sym, I-sym, I-sym,o]
    return:返回的就是实体，疾病类的和症状类的，比如‘肚子疼’
    """

    label_to_index = {"O": 0, "B-dis": 1, "I-dis": 2, "B-sym": 3, "I-sym": 4}
    B_DIS, I_DIS = label_to_index['B-dis'], label_to_index['I-dis']# 1 2
    B_SYM, I_SYM = label_to_index['B-sym'], label_to_index['I-sym']# 3 4

    def extract_word(start_index, next_label):
        # 提取实体
        index, entity = start_index+1, [data_inputs[start_index]] # 注意 有可能不会进入下面的循环，所有需要给index赋值
        # entity = [data_inputs[start_index]]

        for index in range(start_index+1, len(data_labels)):
            if data_labels[index] != next_label:
                #这里的if举个例子，比如我找到一个B-sym开头的实体，以B-sym开头的实体后面一定是I-sym，如果不是的话就证明提取完毕了
                break
            entity.append(data_inputs[index])

        return index, ''.join(entity)#返回的是一个实体：“我肚子疼”


    extract_entities = {'DIS':[], 'SYM': []}#用来保存结果
    index = 0
    next_label = {B_DIS:I_DIS, B_SYM:I_SYM}#{0:1,2:3} 搞这个的作用是为了判断‘肚子疼’这个实体的三个字是否都被提取完毕了
    # print(next_label)

    word_class = {B_DIS:'DIS', B_SYM:'SYM'}#{1:'DIS',2:'SYM'}，它主要用来判断当前获取的实体是属于疾病类的还是症状类的

    while index < len(data_labels): #0 < 5。data_labels是传进来的一句话的标签
        label = data_labels[index] #index=0，data_labels[index]='o'
        if label in next_label.keys():#某一个实体开始出现
            index, word = extract_word(index, next_label[label])#传进来的是一句话实体开始位置的下标和1或者4(1是B-dis对应下标，4是I-sym对应下标)
            extract_entities[word_class[label]].append(word) #提取到的实体有两类，根据lable来判断该存到哪类中
            continue
        index += 1

    return extract_entities



if __name__ == '__main__':
    evaluate()