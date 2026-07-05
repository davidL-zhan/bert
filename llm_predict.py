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
输入 :在当当上买了很多书，都懒于评论。但这套书真的很好，3册都非常精彩。我家小一的女儿，认字多，非常喜爱，每天睡前必读。她还告诉我，学校的语文课本中也有相同的文章。我还借给我的同事的女儿，我同事一直头疼她女儿不爱看书，但这套书，她女儿非常喜欢。两周就看完了。建议买。很少写评论，但忍不住为这套书写下。也给别的读者参考下。
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
