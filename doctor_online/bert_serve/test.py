#todo:5-测试模型部署
import requests

url = "http://0.0.0.0:5001/v1/recognition/"
data = {"text1":"人生该如何起头", "text2": "改变要如何起手"}
res = requests.post(url, data=data)

print("预测样本:", data["text1"], "|", data["text2"])
print("预测结果:", res.text)

'''
执行这段代码前先去Aidoc环境中执行
gunicorn -w 1 -b 0.0.0.0:5001 app:app 
把服务启动
'''