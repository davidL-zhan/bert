import fastapi

from predict import predict
from starlette.responses import JSONResponse
from llm_predict import predict_llm

app = fastapi.FastAPI()
label2answer = {0: "这是一个负面评价", 1: "这是一个正面评价"}


@app.post("/predict")
def predict_text(text: dict):
    """接收文本，返回预测结果"""
    result = predict(text)
    # result = int(predict_llm(text))
    answer = label2answer[result]
    return JSONResponse(content={"label": answer})
