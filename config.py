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


config = Config()
if __name__ == "__main__":
    print(config.model_name)
