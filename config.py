class Config:
    def __init__(self):
        self.dataset_path = "clue/clue"
        self.dataset_name = "tnews"
        self.model_name = "google-bert/bert-base-chinese"
        self.classname_len = 15

        # 训练配置
        self.batch_size = 64
        self.num_workers = 4
        self.learning_rate = 0.001
        self.num_epochs = 10
        self.dropout = 0.1
        self.max_length = 128


config = Config()
if __name__ == "__main__":
    print(config.model_name)
