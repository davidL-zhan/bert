import fastapi
import uvicorn
from predict import predict
from starlette.responses import JSONResponse
from llm_predict import predict_llm
from config import config

app = fastapi.FastAPI()
label2answer = {0: "这是一个负面评价", 1: "这是一个正面评价"}


@app.post("/predict")
def predict_text(text: dict):
    """接收文本，返回预测结果"""
    result = predict(text)
    # result = int(predict_llm(text))
    answer = label2answer[result]
    return JSONResponse(content={"label": answer})


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=config.fastapi_host,
        port=config.fastapi_port,
        reload=config.reload,
    )
