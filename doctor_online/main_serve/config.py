#redis
REDIS_CONFIG = {
     "host": "0.0.0.0",
     "port": 6379
}

#neo4j
NEO4J_CONFIG = {
    "uri": "bolt://0.0.0.0:7687",
    "auth": ("neo4j", "12345678"),
    "encrypted": False
}

model_serve_url = "http://0.0.0.0:5001/v1/recognition/"

TIMEOUT = 2

#用于把疾病实体包装成通顺的句子
reply_path = "./reply.json"

#redis的超时市场
ex_time = 36000#10h
