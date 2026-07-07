from dataset import build_dataloader
from config import config
from model import BertClassifierModel
import torch
import torch.nn as nn
from train import evaluate
import warnings

# warnings.filterwarnings("ignore", category=DeprecationWarning)
"""
        量化引擎：
            none：没有硬件加速
            onednn：英特尔深度学习
            x86：x86架构的服务器进行优化
            fbgemm：FaceBook的量化计算引擎
"""
print("支持的量化引擎：", torch.backends.quantized.supported_engines)


if __name__ == "__main__":
    test_loader = build_dataloader("test", 64, False, 0)
    # print(len(test_loader))
    # print(next(iter(test_loader)))
    model = BertClassifierModel()
    model.load_state_dict(torch.load(r"checkpoints\best.pt", map_location="cpu"))
    result = evaluate(model, test_loader, nn.CrossEntropyLoss(), torch.device("cpu"))
    print(result)
    quantization_model = torch.quantization.quantize_dynamic(
        model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
    )
    quantization_model.eval()
    result = evaluate(
        quantization_model, test_loader, nn.CrossEntropyLoss(), torch.device("cpu")
    )
    print(result)
    torch.save(quantization_model, "checkpoints/quantization_model.pkl")
