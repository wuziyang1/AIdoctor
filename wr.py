#微信公众号服务

import werobot
import requests

url = "http://0.0.0.0:5000/v1/main_serve/"#这个url是;P主逻辑服务的url

# 访问主要逻辑服务时的超时时长
TIMEOUT = 3

robot = werobot.WeRoBot(token="doctoraitoken")#doctoraitoken这是注册公众号的时候设置的

# todo:处理请求入口
#这是微信传进来的
@robot.handler
def doctor(message, sesssion):
    try:
        #todo:1-获取用户的uid
        uid = message.source
        try:
            #todo:2-检查session,判断用户是否是第一次发言
            if sesssion.get(uid, None) != '1':#if条件如果成立就证明该用户是第一次使用公众号
                sesssion[uid] = '1'
                return "您好, 我是智能客服小艾, 有什么需要帮忙的吗?"

            # todo:3-此时用户不是第一次发言,获取用户输入的内容
            text = message.content
        except:
            return "您好, 我是智能客服小艾, 有什么需要帮忙的吗?"

        data = {'uid':uid, 'text':text}
        # todo:4-向主要逻辑服务发送post请求，实现接口时规定是post请求
        res = requests.post(url, data=data, timeout=TIMEOUT)

        return res.text

    except Exception as e:
        print(e)
        return "机器人客服正在休息，请稍后再试..."

robot.config['HOST'] = '0.0.0.0'#设置任何主机都可以访问
robot.config['PORT'] = 80 # 微信平台只会访问80端口
robot.run()