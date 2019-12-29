from transformers import BertConfig


class HyperParameters(object):
    """
    用于管理模型超参数
    """

    def __init__(
            self,
            max_length: int = 128,
            epochs=4,
            batch_size=32,
            learning_rate=2e-5,
            fp16=True,
            fp16_opt_level="O1",
            max_grad_norm=1.0,
            warmup_steps=0.1,
    ) -> None:
        self.max_length = max_length
        """句子的最大长度"""
        self.epochs = epochs
        """训练迭代轮数"""
        self.batch_size = batch_size
        """每个batch的样本数量"""
        self.learning_rate = learning_rate
        """学习率"""
        self.fp16 = fp16
        """是否使用fp16混合精度训练"""
        self.fp16_opt_level = fp16_opt_level
        """用于fp16，Apex AMP优化等级，['O0', 'O1', 'O2', and 'O3']可选，详见https://nvidia.github.io/apex/amp.html"""
        self.max_grad_norm = max_grad_norm
        """最大梯度裁剪"""
        self.warmup_steps = warmup_steps
        """学习率线性预热步数"""

    def __repr__(self) -> str:
        return self.__dict__.__repr__()


class SimMatchModelConfig(BertConfig):
    """
    相似案例匹配模型的配置
    """

    def __init__(self, max_len=512, algorithm="BertForSimMatchModel", **kwargs):
        super(SimMatchModelConfig, self).__init__(**kwargs)
        self.max_len = max_len
        self.algorithm = algorithm
