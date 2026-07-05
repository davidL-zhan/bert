import fastapi

from predict import predict
from starlette.responses import JSONResponse

app = fastapi.FastAPI()


@app.post("/predict")
def predict(text: dict):
    """接收文本，返回预测结果"""
    return JSONResponse(predict(text))

