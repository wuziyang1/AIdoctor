import numpy as np

#todo:这段代码是用于生成数据和前向概率计算的

class HMM(object):
    def __init__(self, N, M, pi=None, A=None, B=None):
        self.N = N # 盒子数量4——状态数量
        self.M = M # 球颜色数量2——观测数量
        self.pi = pi # 初始概率向量每个盒子都是25%的概率
        self.A = A # 转移概率矩阵 从1号盒子到2号盒子....他们怎么转的概率
        self.B = B # 观测概率矩阵 不同的盒子中取到红球的概率分别是多少

    # todo:根据给定的概率分布，返回一个索引
    def get_data_with_distribute(self, dist):
        '''
        对应题目中的：
        开始，从4个盒子里以等概率初始改概率向量25%
        随机选取1个盒子，从这个盒子里随机抽取1个球，记录颜色后放回；
        然后从当前盒子随机转移到下一个盒子，
        规则是：如果当前盒子是盒子1，那么下一盒子一定是盒子2，
        如果当前是盒子2或3，那么分别以概率0.4和0.6转移到左边或右边的盒子，
        如果当前是盒子4，那么各以0.5的概率停留在盒子4或转移到盒子 3
        '''
        return np.random.choice(np.arange(len(dist)), p=dist)
        #dist是一个概率分布数组[0.4, 0, 0.6, 0],这样的话就是返回的0.6对应的索引2，表示的就是从3号盒子取球
        # ,他也不一定返回0.6只是返回0.6的概率大

    #todo:根据给定的参数生成观测序列
    def generate(self, T : int):
        # T 要生成的数据的数量。生成5个颜色，10个颜色

        # todo:根据  初始  概率分布，获取从哪个盒子取第一个球
        z = self.get_data_with_distribute(self.pi) # 得到的是第一个盒子的编号。z就是获取的第一个盒子的编号

        # 从上一个盒子中根据观测概率选中一个球（颜色）
        x = self.get_data_with_distribute(self.B[z]) #x代表球的颜色，0红色 1白色
        result = [x]
        '''
        self.pi题目中规定初始概率四个盒子每个盒子都是25%，假设从第一个盒子中取，z=1
        self.B[z]；取球。这句话就代表在1号盒子中取出红球和白球的概率矩阵[0.5,0.5]
        正好把这个矩阵扔给这个函数get_data_with_distribute，就能返回0或者1.0表示红球1表示白球
        所以get_data_with_distribute这个函数取球的时候用取盒子的时候也用
        '''
        for _ in range(T-1):
            z = self.get_data_with_distribute(self.A[z]) # 根据上一个盒子得到现在这个盒子，比如上一个盒子是1号盒子，根据A这个矩阵就能得到下一个盒子。这里假设是2号盒子
            x = self.get_data_with_distribute(self.B[z]) # 根据现在这个盒子得到从现在这个盒子中取哪个球，然后根据B这个矩阵就能得到从2号盒子中取球取到红球和白球的概率
            result.append(x)

        return result

    #todo:前向概率计算公式
    #就是一致一个序列求出现这个序列的可能的概率，比如已知：红白白白红。求出现这样序列的可能的概率
    def forward_probability(self, X):#X比如是[红,白,红,白,白]
        # 根据给定的观测序列X，计算观测序列出现的概率
        alpha = self.pi * self.B[:, X[0]] #选择某个盒子的初始概率*每个盒子中取红球的概率。X[0]表示红

        for x in X[1:]:
            alpha = np.matmul(alpha, self.A) * self.B[:, x]

        return alpha.sum()


if __name__ == '__main__':

    #todo:初始概率矩阵
    pi = np.array([.25, .25, .25, .25])

    #todo:转移矩阵
    A = np.array([
        [0,  1,  0, 0],
        [.4, 0, .6, 0],
        [0, .4, 0, .6],
        [0, 0, .5, .5]])

    #todo:观测矩阵
    B = np.array([
        [.5, .5],
        [.3, .7],
        [.6, .4],
        [.8, .2]])

    assert len(A) == len(pi)
    assert len(A) == len(B)

    hmm = HMM(B.shape[0], B.shape[1], pi, A, B)
    seq = hmm.generate(5)
    print(seq)  # 生成5个数据

    print(hmm.forward_probability(seq))