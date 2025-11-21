# 导入Flask类
from flask import Flask
# 创建一个该类的实例app, 参数为__name__, 这个参数是必需的，
# 这样Flask才能知道在哪里可找到模板和静态文件等东西。
app = Flask(__name__)

# 使用route()装饰器来告诉Flask触发函数的URL
@app.route('/')
def hello_world():
    """请求指定的url后，执行的主要逻辑函数"""
    # 在用户浏览器中显示信息：'Hello, World!'
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

'''
gunicorn
    使用其启动Flask服务,就是配合flask服务一起使用,减少网络丢包率：
    1. 先来到flask_app.py目录下：
        cd /export/data/AIdoctor/day01
    2.启动服务
        gunicorn -w 1 -b 0.0.0.0:5000 flask_app:app
            -w 代表开启的进程数，我们只开启一个进程
            -b 服务的IP地址和端口
            flask_app:app 是指执行的主要对象位置，在flask_app.py中的app对象
    3.如果使其在后台运行可使用：
        nohup gunicorn -w 1 -b 0.0.0.0:5000 flask_test:app 2>&1 &
    4.成功启动后浏览器访问：
        http://192.168.23.199:5000/

'''