import os


class Config:
    # 集中管理数据、模型、训练、接口和 LLM 相关配置。
    def __init__(self) -> None:
        self.dataset_path: str = "STARRY3056/ChnSentiCorp"
        self.dataset_name: str | None = None
        self.model_name: str = "hfl/chinese-macbert-base"
        # google-bert/bert-base-chinese
        # hfl/chinese-macbert-base
        self.classname_len: int = 2
        self.text_column: str = "text"
        self.label_column: str = "label"

        # 训练配置
        self.batch_size: int = 64
        self.num_workers: int = 4
        self.learning_rate: float = 2e-5
        self.num_epochs: int = 15
        self.dropout: float = 0.1
        self.max_length: int = 128
        self.warmup_ratio: float = 0.1
        self.max_grad_norm: float = 1.0

        # llm
        self.llm_api_key: str | None = os.getenv("DASHSCOPE_API_KEY")
        self.llm_model_name: str = "deepseek-v4-flash"
        self.llm_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

        # fastapi
        self.fastapi_host: str = "127.0.0.1"
        self.fastapi_port: int = 8000
        self.reload: bool = True


# 项目根目录，其他脚本用它拼接 checkpoint、data 等相对路径。
Root_DIR: str = os.path.dirname(os.path.abspath(__file__))

config: Config = Config()
if __name__ == "__main__":
    print(Root_DIR)
    print(config.model_name)
