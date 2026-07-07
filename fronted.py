import streamlit as st
import requests
import time

# 1- 创建streamlit页面
st.title("投满分项目")
st.write("这是一个投满分项目")

# 2- 获得用户输入
text: str = st.text_input("请输入评价")

# 3- 发送请求
if st.button("提交"):
    starttime: float = time.time()

    try:
        # 4- 调用接口
        url: str = "http://127.0.0.1:8000/predict"
        response: requests.Response = requests.post(url, json={"text": text})

        use_time: float = time.time() - starttime

        # 5- 结果解析，展示到页面
        pred_class: str = response.json()["label"]
        st.write(f"耗时：{round(use_time, 3)}s")
        st.write(f"预测结果：{pred_class}")
    except Exception as e:
        # 接口未启动或网络异常时，前端保持原有兜底提示。
        st.write("网络波动，错误码是：666，请联系人工客服：020-119")
# streamlit run fronted.py --server.address 127.0.0.1 --server.port 8501
