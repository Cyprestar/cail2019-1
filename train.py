import logging
import os
import datetime

from model import HyperParameters, BertModelTrainer

logger = logging.getLogger("train model")
logger.setLevel(logging.INFO)
logger.propagate = False
logging.getLogger("transformers").setLevel(logging.ERROR)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

MODEL_DIR = "./output/model" + str(datetime.datetime.now())
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
fh = logging.FileHandler(os.path.join(MODEL_DIR, "train.log"), encoding="utf-8")
fh.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    bert_pretrained_model = './bert/ms'

    training_dataset = './data/raw/CAIL2019-SCM-big/SCM_5k.json'

    valid_input_path = './data/valid/valid.json'
    valid_ground_truth_path = './data/valid/ground_truth.txt'

    test_input_path = "./data/test/test.json"
    test_ground_truth_path = "./data/test/ground_truth.txt"

    config = {
        "max_length": 512,
        "epochs": 10,
        "batch_size": 6,
        "learning_rate": 2e-5,
        "fp16": True,
        "fp16_opt_level": "O1",
        "max_grad_norm": 1.0,
        "warmup_steps": 0.1,
    }
    hyper_parameter = HyperParameters()
    hyper_parameter.__dict__ = config
    algorithm = "BertForSimMatchModel"

    trainer = BertModelTrainer(
        training_dataset,
        bert_pretrained_model,
        hyper_parameter,
        algorithm,
        valid_input_path,
        valid_ground_truth_path,
        test_input_path,
        test_ground_truth_path,
    )
    trainer.train(MODEL_DIR, 1)
