from openai import OpenAI
import os
from config import config

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key=config.llm_api_key,
    base_url=config.llm_url,
)
prompt = """你是一个专业的文本分类器 根据用户输入的评价，返回评价的分类结果
分类结果只有0和1 0表示差评 1表示好评
例如
输入 :怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片！开始还怀疑是不是赠送的个别现象，可是后来发现每张DVD后面都有！真不知道生产商怎么想的，我想看的是猫和老鼠，不是米老鼠！如果厂家是想赠送的话，那就全套米老鼠和唐老鸭都赠送，只在每张DVD后面添加一集算什么？？简直是画蛇添足！！
输出 :1
输入 :19天硬盘就罢工了~~~算上运来的一周都没用上15天~~~可就是不能换了~~~唉~~~~你说这算什么事呀~~~
输出 :0
严格限制输出 不要任何解释 只能根据输入 输出0或1


"""


def predict_llm(text: dict) -> str | None:
    response = client.chat.completions.create(
        model=config.llm_model_name,
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": text["text"],
            },
        ],
    )
    res = response.choices[0].message.content
    return res


if __name__ == "__main__":
    res = predict_llm({"text": "我非常喜欢这个电影"})
    print(res)
    print(type(res))
