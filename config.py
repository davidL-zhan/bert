import os


class Config:
    def __init__(self):
        self.dataset_path = "STARRY3056/ChnSentiCorp"
        self.dataset_name = None
        self.model_name = "hfl/chinese-macbert-base"
        # google-bert/bert-base-chinese
        # hfl/chinese-macbert-base
        self.classname_len = 2
        self.text_column = "text"
        self.label_column = "label"
        # 训练配置
        self.batch_size = 64
        self.num_workers = 4
        self.learning_rate = 2e-5
        self.num_epochs = 15
        self.dropout = 0.1
        self.max_length = 128
        self.warmup_ratio = 0.1
        self.max_grad_norm = 1.0

        # llm
        self.llm_api_key = os.getenv("DASHSCOPE_API_KEY")
        self.llm_model_name = "deepseek-v4-flash"
        self.llm_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        # fastapi
        self.fastapi_host = "127.0.0.1"
        self.fastapi_port = 8000
        self.reload = True


Root_DIR = os.path.dirname(os.path.abspath(__file__))

config = Config()
if __name__ == "__main__":
    print(Root_DIR)
    print(config.model_name)
