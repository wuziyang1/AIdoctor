#todo:01-构建NER模型
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, label_num):
        super(BiLSTM, self).__init__()

        # embeding字嵌入
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=256,device=device)

        # bilstm，没有BiLSTM对象，只有LSTM，bidirectional
        self.blstm = nn.LSTM(input_size=256,hidden_size=512,bidirectional=True,num_layers=1)

        # 线性层, 最终输出是发射概率矩阵
        self.linear = nn.Linear(in_features=1024, out_features=label_num)#正向512反向512整合后1024
        # label_num：{"O": 0, "B-dis": 1, "I-dis": 2, "B-sym": 3, "I-sym": 4}共5个
        #这里输出的就是观测概率矩阵：就是把一句话转BMSE这种用单词表示的形式

    def forward(self, inputs, length):
        # 嵌入层，得到向量
        outputs_embed = self.embed(inputs)#[batch_size,seq_len,num_features]

        # 打padding
        outputs_packd = pack_padded_sequence(outputs_embed, length.cpu())

        # 把压缩后的结果输入到lstm中
        outputs_blstm, (hn, cn) = self.blstm(outputs_packd)

        # blstm训练完再把padding还原
        outputs_paded, outputs_lengths = pad_packed_sequence(outputs_blstm)

        # 调整形状，batch_size放在下标为0的维度
        outputs_paded = outputs_paded.transpose(0, 1)#把batch放的第0维度

        # 线性层
        outputs_logits = self.linear(outputs_paded)#outputs_logits就是一句话的发射概率矩阵

        # 取出每句话真实长度发射概率矩阵
        # 最终输入到crf 作为emission_score
        outputs = []

        for outputs_logit, outputs_length in zip(outputs_logits, outputs_lengths):#为啥要循环，因为一个batch中有多个句子
            outputs.append(outputs_logit[:outputs_length])#把这句话除了padding外的真实的部分加入outputs。

        return outputs

    def predict(self, inputs):
        output_embed = self.embed(inputs)
        # print('output_embed.shape:', output_embed.shape)

        # 在batch size增加一个维度1
        output_embed = output_embed.unsqueeze(1)
        # print('output_embed.shape1:', output_embed.shape)

        output_blstm, (hn, cn) = self.blstm(output_embed)

        output_blstm = output_blstm.squeeze(1)

        output_linear = self.linear(output_blstm)

        return output_linear
#这里可以直接用BiLSTM的输出结果作为答案，但是为什么不用呢
#因为CRF可以学习一些先验知识，比如一句话开头不能是E（BMSE）


class CRF(nn.Module):
    def __init__(self, label_num):
        super(CRF, self).__init__()

        # todo:定义标签数量
        self.label_num  = label_num#状态的数量,B-Person, I-Person, B-Organization, I-Organization, O

        # todo:随机初始化转移分数，模型需要学习的参数
        self.transition_scores = nn.Parameter(torch.randn(self.label_num+2, self.label_num+2))
        # +2原因，在B-Person, I-Person, B-Organization, I-Organization, O基础上增加了START_TAG END_TAG
        #转移概率矩阵是CRF模型学习的重要参数，他是通过nn.Parameter定义的，后续会通过optimizer.step()自动更新


        # 设置start_tag end_tag的编号
        self.START_TAG, self.END_TAG = self.label_num, self.label_num+1

        # todo:初始化start end--保证其他状态不会转到START 到END之后不会再转到其他状态
        # 定义填充的常量
        self.fill_value = -1000
        self.transition_scores.data[:, self.START_TAG] = self.fill_value#这里设置成-1000怎么计算都永远不可能是最大值
        self.transition_scores.data[self.END_TAG, :] = self.fill_value#


    # todo:1-计算单条路径的分数Sreal
    def _get_real_path_score(self, emission_score, sequence_label):
        # emission_score是BilSTM输出结果
        # sequence_label自己标注的真实的序列标签

        # 前一时刻到后一时刻转换的时候的累加

        # 保存sequence长度
        seq_len = len(sequence_label)#序列长度，多少个字

        # todo:1-计算发射分数
        # emission_score中每一行是一个字的发射分数，就是发射成B-Person, I-Person, B-Organization, I-Organization, O各自的概率
        real_emission_score = torch.sum(emission_score[list(range(seq_len)), sequence_label])#[第i行,第j列]
        # wzy:每一时刻发射到对应标签上的概率值累加
        # 这里是拿着真实标签sequence_label去训练得来的emission_score这个矩阵中找真实标签对应的分数

        #todo:2-计算转移路径分数
        #todo:2-1先把原序列加上 start 和 end
        #[1, 2, 3] sequence_label
        b_id = torch.tensor([self.START_TAG], dtype=torch.int32, device=device)
        e_id = torch.tensor([self.END_TAG], dtype=torch.int32,  device=device)
        sequence_label_expand = torch.cat([b_id, sequence_label, e_id])
        # 再真实标签前后增加 start end
        # 结果类似 [5, 1, 2, 3, 6]

        # todo:2-2获取转移路径
        pre_tag = sequence_label_expand[list(range(seq_len+1))]
        # 结果类似 [5, 1, 2, 3]
        now_tag = sequence_label_expand[list(range(1, seq_len+2))]
        # 结果类似 [1, 2, 3, 6]
        real_transition_score = torch.sum(self.transition_scores[pre_tag, now_tag])
        # 这样错开一位构造十分巧妙，表示从前一个状态转移到后一个状态的转移值
        # 去转移概率矩阵中就可以拿到这个值了

        # todo:3-真实路径分数 = 发射分数 + 转移路径分数
        real_path_score = real_emission_score + real_transition_score

        return real_path_score

    #todo:2-2计算log_exp值
    def _log_sum_exp(self, score):
        # 这部分的公式就是pdf第7页最上面的那个
        # 根据公式计算路径的分数
        # 为了避免计算时溢出，每个元素先减去最大值，计算完成后，再把最大值加回来
        max_score, _ = torch.max(score, dim=0)
        max_score_expand = max_score.expand(score.shape)#每个元素都减去max值
        return max_score + torch.log(torch.sum(torch.exp(score-max_score_expand)))

    #todo:2-1扩展发射概率矩阵增加两行两列
    # 两行指的是在一句话开头增加的start字符结尾增加的end字符
    # 两列指的是开始的tag和结束的tag
    def _expand_emission_matrix(self, emission_score):  #emission_score:[seq_len,tag_len][3,5]
        # 对发射分数进行扩充，因为添加了start end两个标签
        # emission_score的形状
        # [字的个数, 5] 5代表的是 len [B-dis I-dis B-sym I-sym O]
        # 获取序列长度
        # 比如emission_score对应的是 我头疼 的发射分数矩阵
        # 是 3 * 5矩阵
        seq_length = emission_score.shape[0]

        # todo:前期准备1--增加start end这两个标签
        b_s = torch.tensor([[self.fill_value] * self.label_num + [0, self.fill_value]],device=device)
        # b_s e_s 都是1 * 7向量,b_s这7个数分别是[-1000,-1000,-1000,-1000,-1000,0,-1000]
        #设置成-1000表示不可能从别的标签转移的start或者end，因为他俩没有任何意义，0表示start可以转移的自己

        e_s = torch.tensor([[self.fill_value] * self.label_num + [self.fill_value, 0]],device=device)
        #e_s : [-1000,-1000,-1000,-1000,-1000,-1000,0]

        # todo:前期准备2--先初始化
        expand_matrix = self.fill_value * torch.ones([seq_length, 2], dtype=torch.float32, device=device)
        # 3 * 2 值全为-1000

        # todo:1-[3,5]->[3,7]加两列
        emission_score_expand = torch.cat([emission_score, expand_matrix], dim=1)# 3 * 7

        # todo:2-[3,7]->[5,7]加两行
        emission_score_expand = torch.cat([b_s, emission_score_expand, e_s], dim=0)# 5 * 7

        return emission_score_expand

    #todo:2-获取全部路径分数
    def _get_total_path_score(self, emission_score):

        # todo:1- 扩展发射分数矩阵
        emission_score_expand = self._expand_emission_matrix(emission_score)

        # 计算所有路径分数
        pre = emission_score_expand[0] # pre代表的是累计到上一个时刻，每个状态之前的所有路径分数之和,相当于课件上的alpha
        # pre的形状是[1*7]，初始时刻表示第一个字的7个发射分数
        for obs in emission_score_expand[1:]:#从序列的第二个词开始处理每个词，obs表示当前词的发射分数
            # todo:1-扩展pre的维度，把pre转置，横向广播一个维度，pre就是课件中的alpha
            pre_expand = pre.reshape(-1, 1).expand([self.label_num+2, self.label_num+2])
            # pre本来是一行，reshape后变成一列[标签数,1]，expand后变成[标签数+2,标签数+2]，每一行内容相同，其值代表了上一时间步所有状态的累积分数。

            # todo:2-扩展obs的维度，纵向添加一个维度
            obs_expand = obs.expand([self.label_num+2, self.label_num+2])
            #obs是一行,[1,标签数]这里扩展成[标签数+2,标签数+2]其实就是复制，其值代表了当前词到各个标签的分数。


            # todo:3-按照矩阵计算的目录，计算上一个时刻的每种状态 到这个时刻的每种状态的组合方式全部包含在矩阵运算
            score = obs_expand + pre_expand + self.transition_scores#transition_scores不用变
            #该时刻的发射分数 + 累积到该时刻前所有时刻的总分数 + 上一时刻转移到该时刻的转移分数矩阵
            #score矩阵包含了从上一步所有状态到当前步所有状态的所有可能组合的分数。

            # todo:4-计算分数,就是对第三步的结果进行log_sum_exp
            # print('\nscore:', score)
            # print('\nscore.shape:', score.shape)
            pre = self._log_sum_exp(score)
            # 1 x 7 每一列代表的是上一个时刻的所有状态到这个时刻的某一个状态之和
        # for结束仍然得到一个pre 代表是最后一个时刻, 1 x 7 每一列代表的是上一个时刻的所有状态到这个时刻的某一个状态之和

        # 因为for循环执行完成后，pre最后一个时刻，也就是每个状态之前的所有路径之和
        # 最终结果计算全部路径之和，因此还需要进行最后一步计算
        return self._log_sum_exp(pre)

    def forward(self, emission_scores, sequence_labels):
        # todo:计算损失值
        # 是一个批次的
        total = 0.0
        for emission_score, sequence_label in zip(emission_scores, sequence_labels):
            real_path_score = self._get_real_path_score(emission_score, sequence_label)
            total_path_score = self._get_total_path_score(emission_score)
            loss = total_path_score - real_path_score
            total += loss

        return total

    #todo:3-使用维特比算法进行预测
    def predict(self, emission_score):
        # todo:1-扩展emission_score-就是把start end加到矩阵中去
        emission_score_expand = self._expand_emission_matrix(emission_score)

        #todo:2- 记录每个时刻对应每个状态对应的 最大分数，以及索引
        ids = torch.zeros(1, self.label_num+2, dtype=torch.long, device=device)
        val = torch.zeros(1, self.label_num+2, device=device)

        pre = emission_score_expand[0]

        for obs in emission_score_expand[1:]:
            # 对pre进行旋转
            pre_extend = pre.reshape(-1, 1).expand([self.label_num+2, self.label_num+2])
            obs_extend = obs.expand([self.label_num+2, self.label_num+2])

            # 累加，矩阵对用位置进行累加，得到的结果是上一个时刻的所有状态到这个时刻的所有状态可能转移方式
            score = obs_extend + pre_extend + self.transition_scores

            # todo:不同点 -- 记录当前时刻最大的分值和索引
            value, index = score.max(dim=0)#取出每一个字的最大得分和对应的索引
            ids = torch.cat([ids, index.unsqueeze(0)], dim=0)#保存的是上一步的index和这一步的index[seq_len,2]
            val = torch.cat([val, value.unsqueeze(0)], dim=0)
            #wzy：维特比算法和前向算法最大的区别就是维特比这里记录了最大值的下标，仅仅就是这一点差别

            pre = value

        # todo:取出最后一个时刻的最大值
        index = torch.argmax(val[-1])
        best_path = [index]
        # print('val[-1]:', val[-1])
        # print('best_path:', best_path)

        for i in reversed(ids[1:]):#reversed反转从后往前找
            index = i[index].item()#根据上一步的index找到这一步的index
            best_path.append(index)
            # print(i, 'best_path:', best_path)

        best_path = best_path[::-1][1:-1]#再给他反转回来

        return best_path


#todo:3-把CRF和BilSTM拼起来
class NER(nn.Module):
    def __init__(self, vocab_size, label_num):
        super(NER, self).__init__()

        self.vocab_size = vocab_size#seq_len
        self.label_num = label_num #状态的数量tag0,tag1,tag2,tag3

        self.bilstm = BiLSTM(vocab_size=self.vocab_size, label_num=self.label_num)

        self.crf = CRF(label_num=self.label_num)

    def forward(self, inputs, labels, length):
        #todo:1- bilstm的forward函数返回 发射分数矩阵
        emission_scores = self.bilstm(inputs, length)#length=MAX_LEN
        #todo:2- 得到一个批次的损失值
        batch_loss = self.crf(emission_scores, labels)

        return batch_loss

    def predict(self, inputs):
        # 预测
        # 得到输入句子的发射分数矩阵
        # print('inputs.shape:', inputs.shape)
        emission_scores = self.bilstm.predict(inputs)
        logits = self.crf.predict(emission_scores)

        return logits

    def save_model(self, save_path):
        save_info = {
            'init': {'vocab_size': self.vocab_size, 'label_num': self.label_num},
            'state': self.state_dict()
        }
        torch.save(save_info, save_path)




if __name__ == '__main__':
    char_to_id = {"双": 0, "肺": 1, "见": 2, "多": 3, "发": 4, "斑": 5, "片": 6,
                  "状": 7, "稍": 8, "高": 9, "密": 10, "度": 11, "影": 12, "。": 13}

    # 参数2:标签码表对照
    tag_to_id = {"O": 0, "B-dis": 1, "I-dis": 2, "B-sym": 3, "I-sym": 4}
    bilstm = BiLSTM(vocab_size=len(char_to_id),
               label_num=len(tag_to_id),)
    print(bilstm)
