'''
todo:任务是把完成命名实体审核后的结构化数据写入neo4j
'''

# 导包
import os
import fileinput
from neo4j import GraphDatabase

# 设置neo4j图数据库的配置信息
NEO4J_CONFIG = {
	"uri": "neo4j://192.168.23.199:7687",
	"auth": ("neo4j", "12345678"),
	"encrypted": False
}

driver = GraphDatabase.driver(**NEO4J_CONFIG) # 两个星号对字典进行解包，key就是参数名

# todo:1 读取数据
def _load_data(path):
    """
    先获取文件名称，也就是疾病名称
    获取每个文件的内容，也就是症状
    构造成字典：{疾病1: [症状1, 症状2, ...], 疾病2: [症状1, 症状2, ...]}
    :param path: 文件存储的路径
    :return: 构造的字典
    """
    # 获取疾病csv列表
    disesase_csv_list = os.listdir(path)

    # 去掉.csv后缀，得到疾病列表 xxx.csv
    disesase_list = list(map(lambda x: x.split('.')[0], disesase_csv_list))

    # 初始化一个列表 存放每种疾病对应的症状的列表
    symptom_list = []

    # 遍历每个csv
    for disesase_csv in disesase_csv_list:
        symptom = list(map(lambda x:x.strip(), fileinput.FileInput(os.path.join(path, disesase_csv))))
        symptom = list(filter(lambda x: 0 < len(x) < 100, symptom))
        symptom_list.append(symptom)

    # 转成字典，返回
    return dict(zip(disesase_list, symptom_list))


# todo:2 把数据写入到neo4j中
def neo4j_write(path):
    """
    把数据写入到neo4j中
    :param path: 存放的是通过审核模型审核完成的疾病的名称和对应的症状文件
    :return:
    """
    # 保存上一步返回的字典
    disease_symptom_dict = _load_data(path)

    # 开启会话，写入数据到neo4j中
    with driver.session() as session:
        for key, values in disease_symptom_dict.items():
            # key 就是疾病名称
            # values 是一个列表，key这个疾病对应的多种症状

            # 创建疾病节点
            cypher = "MERGE (a:Disease{name:%r}) RETURN a" % key #wzy:把一种疾病放到一个表中，n中疾病也就是n个表
            session.run(cypher)

            for value in values:#遍历这一种疾病的所有症状
                # 创建症状节点
                cypher = "MERGE (b:Symptom{name:%r}) RETURN b" % value #wzy:把一种症状存到一个表。使用MERGE创建下次再遇到相同症状就不会重复创建了
                session.run(cypher)

                # 创建疾病和症状的关系
                cypher = "MATCH (a:Disease{name:%r}) MATCH(b:Symptom{name:%r}) WITH a, b MERGE(a)-[r:dis_to_sym]-(b)" % (key, value)
                session.run(cypher)

        # 适用于 Neo4j 4.x 及以上的索引创建语法
        cypher = "CREATE INDEX disease_name_index IF NOT EXISTS FOR (d:Disease) ON (d.name)"
        session.run(cypher)
        cypher = "CREATE INDEX symptom_name_index IF NOT EXISTS FOR (s:Symptom) ON (s.name)"
        session.run(cypher)

        '''
        报错代码：
        #创建两个索引
        cypher = "CREATE INDEX ON:Disease(name)"
        session.run(cypher)
        cypher = "CREATE INDEX ON:Symptom(name)"
        session.run(cypher)
        '''


if __name__ == '__main__':
    neo4j_write('data/structured/reviewed/') # 使用相对路径




'''
命名实体审核：
    这一步是为了过滤掉那些不符合医学描述的症状词语，只留下规范的症状描述词
    
    其实就是短文本二分类问题(RNN)
    
    因为数据集较小，自己完成文本数值化张量化效果不好，所以直接使用BERT预训练模型
    
    症状名词==》bert中得到每个字的字向量==》然后把子向量==》RNN中判断是不是属于命名实体
    
训练RNN时的数据集：
    正样本不变
    负样本其实就是正样本翻转后的结果


'''