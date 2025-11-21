import json
import random
import requests

# client_id 为官网获取的AK， client_secret 为官网获取的SK
client_id = "74b8zns7KLC9T4Fv9UA0Fwra"
client_secret = "9kEGPhgTdlliXXEbZYkoDUbDjyBiBMIZ"
# service_id是机器人id
service_id = "S70282"

def unit_chat(chat_input, terminal_id="88888"):
    """
    description:调用百度UNIT接口，回复聊天内容
    Parameters
      ----------
      chat_input : str
          用户发送天内容
      terminal_id : str
          发起聊天用户ID，可任意定义
    Return
      ----------
      返回unit回复内容
    """
    # todo:设置默认回复内容， 一旦接口出现异常，回复该内容
    chat_reply = "不好意思，俺们正在学习中，随后回复你。"
    # 根据 client_id 与 client_secret 获取access_token
    url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=%s&client_secret=%s" % (client_id, client_secret)

    #todo:发送求情获取token
    res = requests.get(url)
    access_token = eval(res.text)["access_token"]

    #todo: 根据 access_token 获取聊天机器人接口数据
    unit_chatbot_url = "https://aip.baidubce.com/rpc/2.0/unit/service/v3/chat?access_token=" + access_token

    #todo: 拼装聊天接口对应请求发送数据，主要是填充 query 值
    post_data = {
                "log_id": str(random.random()),
                "request": {
                    "query": chat_input,
                    "terminal_id":terminal_id
                },
                "session_id": "",
                "service_id": service_id,
                "version": "3.0"
            }

    # todo:将封装好的数据作为请求内容，发送给Unit聊天机器人接口，并得到返回结果
    res = requests.post(url=unit_chatbot_url, json=post_data)

    # todo:获取聊天接口返回数据
    unit_chat_obj = json.loads(res.content)

    # todo:判断聊天接口返回数据是否出错 error_code == 0 则表示请求正确
    if unit_chat_obj["error_code"] != 0: return chat_reply

    # todo:解析聊天接口返回数据，找到返回文本内容 result -> responses -> schema -> intent_confidence(>0) -> actions -> say
    unit_chat_obj_result = unit_chat_obj["result"]
    unit_chat_response_list = unit_chat_obj_result["responses"]

    # todo:随机选取一个"意图置信度"[+responses[].schema.intent_confidence]不为0的作为回答
    unit_chat_response_obj = random.choice(
       [unit_chat_response for unit_chat_response in unit_chat_response_list if#列表生成式
        unit_chat_response["schema"]["intents"][0]["intent_confidence"] > 0.0])

    # todo:获取所有答复，并随机选择一个
    unit_chat_response_action_list = unit_chat_response_obj["actions"]
    unit_chat_response_action_obj = random.choice(unit_chat_response_action_list)
    unit_chat_response_say = unit_chat_response_action_obj["say"]
    return unit_chat_response_say


if __name__ == '__main__':
    while True:
        chat_input = input("请输入：")
        if chat_input == 'Q' or chat_input == 'q' or chat_input == 'bye':
            break
        print(chat_input)
        chat_reply = unit_chat(chat_input)
        print("用户输入 >>>", chat_input)
        print("Unit回复 >>>", chat_reply)
