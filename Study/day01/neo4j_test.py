from neo4j import GraphDatabase

uri = "neo4j://192.168.23.199:7687"

driver = GraphDatabase.driver(uri, auth=("neo4j", "12345678"))


# 创建一个会话
with driver.session() as session:
    cypher = "create (c:Company) SET c.name='吴紫阳' return c.name"
    record = session.run(cypher) #执行
    print(type(record), record) #, list(record))
    result = list(map(lambda x: x[0], record))
    print("result: ", result)




#todo:neo4j的事务性
#下面可以正常执行
def _some_operations(sess, cat_name, mouse_name):
    sess.run("MERGE (a:Cat{name: $cat_name})"
           "MERGE (b:Mouse{name: $mouse_name})"
           "MERGE (a)-[r:And]-(b)",
           cat_name=cat_name, mouse_name=mouse_name)


with driver.session() as session:
    session.write_transaction(_some_operations, "Tom", "Jerry")

# 下面执行时报错
def _some_operations(sess, cat_name, mouse_name):
    sess.run("MERGE (a:Cat{name: $cat_name})"
           "MERGE (b:Mouse{name: $mouse_name})"
           "CREATE (a)-[r:And]-(b)",
           cat_name=cat_name, mouse_name=mouse_name)

with driver.session() as session:
    session.write_transaction(_some_operations, "Tom1", "Jerry1")


# 就是要么一起成功要么一起失败
#     即使前面成功了，后面有一个失败的话前面成功的也会回退


'''
CREATE (e:Employee{id:222, name:'Bob', salary:6000, deptnp:12})
    1.e ：节点变量：就像编程语言中的变量一样，用来在查询中引用和操作该节点。
        在执行完这条sql语句后这个变量就失效了。起作用的其实还是标签，
    2.Employee：标签：数据库中的表名 ，如果两个节点颜色一样就表示这两个节点在一个表中，是同一个表中的两行
    3.{id:222, name:'Bob', salary:6000, deptnp:12}：属性：表中存的数据

2.数据库索引
    先根据原始表建立一个索引表，为了是查找的时候快速查找
    比如原始表是一个班所有人的信息，索引表就可以是男生和女生分开，
    CREATE INDEX ON:Employee(id)根据id这一列创建一个索引表
'''