import redis
# redis配置
REDIS_CONFIG = {
     "host": "0.0.0.0",
     "port": 6379
}

# 创建一个redis连接池
pool = redis.ConnectionPool(**REDIS_CONFIG)

# 从连接池中初始化一个活跃的连接对象
r = redis.StrictRedis(connection_pool=pool)

uid = "8888"# uid代表某个用户的唯一标识

key = "该用户最后一次说的话：".encode('utf-8')# key是需要记录的数据描述

value = "再见，董小姐".encode('utf-8')# value是需要记录的数据具体内容
r.hset(uid, key, value)

result = r.hget(uid, key)# hget表示使用hash数据结构进行数据读取
print(result.decode('utf-8'))


'''
1. 启动redis-server, 这里使用了默认配置，端口是6379.
    redis-server
2.停止redis服务：
    kill -9 6379

'''