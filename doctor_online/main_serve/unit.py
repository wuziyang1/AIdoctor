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
    连接百度闲聊机器人
    :param chat_input: 用户输入的内容
    :param terminal_id:  用户ID
    :return: 返回unit闲聊内容
    """
    # 1 获取access token
    # 设置默认回复内容,  一旦接口出现异常, 回复该内容
    chat_reply = "不好意思，俺们正在学习中，随后回复你。"
    # 根据 client_id 与 client_secret 获取access_token
    url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=%s&client_secret=%s" % (
    client_id, client_secret)

    # 发送请求，获取token
    res = requests.get(url)
    print(res.text)
    access_token = eval(res.text)['access_token']
    print('access_toke', access_token)

    # 2 根据acess token 获取闲聊内容
    unit_chatbot_url = "https://aip.baidubce.com/rpc/2.0/unit/service/v3/chat?access_token=" + access_token
    # 拼接data数据
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

    # 发送post请求
    res = requests.post(url=unit_chatbot_url, json=post_data)
    # 获取聊天接口返回数据
    unit_chat_obj = json.loads(res.content)
    print(unit_chat_obj)

    # 3 选中unit返回结果，返回给用户
    # 判断返回结果是否正常
    print(unit_chat_obj['error_code'])
    if unit_chat_obj['error_code'] != 0: return chat_reply

    # 解析unit返回的结果
    print(unit_chat_obj['result']['responses'])
    unit_chat_responses_list = unit_chat_obj['result']['responses']

    unit_chat_responses_obj = random.choice(
        [
            unit_chat_response for unit_chat_response in unit_chat_responses_list if unit_chat_response['schema']['intents'][0]['intent_confidence'] > 0
        ]
    )
    print(unit_chat_responses_obj)

    # 随机选中一个答复
    unit_chat_response_action_list = unit_chat_responses_obj['actions']
    print(unit_chat_response_action_list)

    unit_chat_response_action_obj = random.choice(unit_chat_response_action_list)

    return unit_chat_response_action_obj["say"]



if __name__ == '__main__':
    # print(unit_chat('你好'))
    while True:
        chat_input = input("请输入:")
        if chat_input == 'Q' or chat_input == 'q' or chat_input == 'bye':
            break
        print(chat_input)
        chat_reply = unit_chat(chat_input)
        print("用户输入 >>>", chat_input)
        print("Unit回复 >>>", chat_reply)