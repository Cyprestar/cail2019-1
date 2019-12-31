import json
import logging
import os
import random
from typing import Tuple, List, Union

import numpy as np
import torch
from sklearn.model_selection import KFold

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    BertTokenizer,
)

from config import HyperParameters, SimMatchModelConfig
from data import TripletTextDataset, get_collator
from models.bert_esim import BertForSimMatchModel

logger = logging.getLogger("train model")

algorithm_map = {"BertForSimMatchModel": BertForSimMatchModel}


class BertSimMatchModel(object):
    """
    基于 Bert 实现的案件相似匹配模型
    """

    def __init__(self, model, tokenizer, config: SimMatchModelConfig, device: torch.device = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = config.max_len
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else device
        )
        self.model.to(self.device)
        self.model.eval()
        self.algorithm = config.algorithm
        self.model_class = algorithm_map[self.algorithm]
        self.predict_batch_size = 8

    def save(self, model_dir):
        """
        存储模型

        :param model_dir:
        :return:
        """
        # Save a trained model, configuration and tokenizer
        model_to_save = (self.model.module if hasattr(self.model, "module") else self.model)  # Only save the model it-self
        model_to_save.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)

    @classmethod
    def load(cls, model_dir, device=None):
        """
        加载模型。通过模型文件构造实例

        :param model_dir:
        :param device:
        :return:
        """
        tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=False)
        config = SimMatchModelConfig.from_pretrained(model_dir)
        model_class = algorithm_map[config.algorithm]
        model = model_class.from_pretrained(model_dir)
        return cls(model, tokenizer, model.config, device)

    def predict(self, text_tuples: Union[List[Tuple[str, str, str]], TripletTextDataset]) -> List[Tuple[str, float]]:
        if isinstance(text_tuples, Dataset):
            data = text_tuples
        else:
            text_a_list, text_b_list, text_c_list = [list(i) for i in zip(*text_tuples)]
            data = TripletTextDataset(text_a_list, text_b_list, text_c_list, None)

        sampler = SequentialSampler(data)
        collate_fn = get_collator(self.max_length, self.device, self.tokenizer)
        dataloader = DataLoader(data, sampler=sampler, batch_size=8, collate_fn=collate_fn)

        final_results = []

        steps = tqdm(dataloader)

        for step, batch in enumerate(steps):
            with torch.no_grad():
                predict_results = self.model(*batch, mode="prob").cpu().numpy()
                cata_indexes = np.argmax(predict_results, axis=1)

                for i_sample, cata_index in enumerate(cata_indexes):
                    prob = predict_results[i_sample][cata_index]
                    label = "B" if cata_index == 0 else "C"
                    final_results.append((str(label), float(prob)))

        return final_results


class BertModelTrainer(object):
    def __init__(
            self,
            dataset_path,
            bert_model_dir,
            param: HyperParameters,
            algorithm,
            test_input_path,
            test_ground_truth_path,
    ) -> None:
        """

        :param dataset_path: 数据集路径。 默认当作是训练集，但当train函数采用了kfold参数时，将对该数据集进行划分并做交叉验证
        :param bert_model_dir: 预训练 bert 模型路径
        :param param: 超参数
        :param algorithm: 选择算法，默认 BertForSimMatchModel
        :param test_input_path: 固定的测试集的路径，用于快速测试模型性能
        :param test_ground_truth_path: 固定的测试集的标记
        """
        self.dataset_path = dataset_path
        self.bert_model_dir = bert_model_dir
        self.param = param
        self.test_input_path = test_input_path
        self.test_ground_truth_path = test_ground_truth_path
        self.algorithm = algorithm
        self.model_class = algorithm_map[self.algorithm]
        logger.info("Algorithm: " + algorithm)

    def load_dataset(self, n_splits: int = 1) -> List[Tuple[TripletTextDataset, TripletTextDataset, List[str]]]:
        """
        划分k折交叉验证数据集用于cv

        :param n_splits:
        :return: List[(train_data, test_data, test_labels_list)]
        """

        data = []

        if n_splits == 1:
            train_data = TripletTextDataset.from_jsons(self.dataset_path, use_augment=True)
            test_data = TripletTextDataset.from_jsons(self.test_input_path)
            with open(self.test_ground_truth_path) as f:
                test_label_list = [line.strip() for line in f.readlines()]

            data.append((train_data, test_data, test_label_list))
            return data

        raw_data_list = []
        with open(self.dataset_path, encoding="utf-8") as raw_input:
            for line in raw_input:
                raw_data_list.append(json.loads(line.strip(), encoding="utf-8"))

        kf = KFold(n_splits, shuffle=True, random_state=42)
        random.seed(42)
        for train_index, test_index in kf.split(raw_data_list):
            # 准备训练集
            train_data_list = [raw_data_list[i] for i in train_index]
            train_data = TripletTextDataset.from_dict_list(
                train_data_list, use_augment=True
            )

            # 准备测试集，打乱BC顺序
            test_data_list = [raw_data_list[i] for i in test_index]
            shuffled_test_data_list = []
            test_label_list = []
            for item in test_data_list:
                a = item["A"]
                b = item["B"]
                c = item["C"]

                choice = int(random.getrandbits(1))
                label = "B" if choice == 0 else "C"
                if label == "C":
                    item = {"A": a, "B": c, "C": b}

                shuffled_test_data_list.append(item)
                test_label_list.append(label)

            test_data = TripletTextDataset.from_dict_list(shuffled_test_data_list)

            data.append((train_data, test_data, test_label_list))
        return data

    def train(self, model_dir, kfold=1):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        logger.info("***** Running training *****")
        logger.info("dataset: {}".format(self.dataset_path))
        logger.info("k-fold number: {}".format(kfold))
        logger.info("device: {} n_gpu: {}".format(device, n_gpu))
        logger.info(
            "config: {}".format(
                json.dumps(self.param.__dict__, indent=4, sort_keys=True)
            )
        )

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(42)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        tokenizer = BertTokenizer.from_pretrained(self.bert_model_dir, do_lower_case=True)
        data = self.load_dataset(kfold)

        all_acc_list = []
        for k, (train_data, test_data, test_label_list) in enumerate(data, start=1):
            one_fold_acc_list = []
            bert_model = self.model_class.from_pretrained(self.bert_model_dir, output_hidden_states=True)
            bert_model.to(device)

            config = bert_model.config
            config.max_len = self.param.max_length
            config.algorithm = self.algorithm

            num_train_optimization_steps = (int(len(train_data) / self.param.batch_size) * self.param.epochs)

            param_optimizer = list(bert_model.named_parameters())
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    "weight_decay": 0.01,
                },
                {
                    "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

            if self.param.warmup_steps < 1:
                num_warmup_steps = (num_train_optimization_steps * self.param.warmup_steps)
            else:
                num_warmup_steps = self.param.warmup_steps
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.param.learning_rate, eps=1e-8)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_train_optimization_steps,
            )

            if self.param.fp16:
                try:
                    from apex import amp
                except ImportError:
                    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

                bert_model, optimizer = amp.initialize(bert_model, optimizer, opt_level=self.param.fp16_opt_level)

            if n_gpu > 1:
                bert_model = torch.nn.DataParallel(bert_model)

            global_step = 0
            bert_model.zero_grad()

            logger.info("***** fold {}/{} *****".format(k, kfold))
            logger.info("  Num examples = %d", len(train_data))
            logger.info("  Batch size = %d", self.param.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps)

            train_sampler = RandomSampler(train_data)

            collate_fn = get_collator(self.param.max_length, device, tokenizer)

            train_dataloader = DataLoader(
                dataset=train_data,
                sampler=train_sampler,
                batch_size=self.param.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn,
                drop_last=True,
            )
            bert_model.train()
            for epoch in range(int(self.param.epochs)):
                tr_loss = 0
                steps = tqdm(train_dataloader)
                for step, batch in enumerate(steps):
                    # if step % 200 == 0:
                    #     model = BertSimMatchModel(bert_model, tokenizer, self.param.max_length, self.algorithm)
                    #     acc, loss = self.evaluate(model, test_data, test_label_list)
                    #     logger.info(
                    #         "Epoch {}, step {}/{}, train Loss: {:.7f}, eval acc: {}, eval loss: {:.7f}".format(
                    #             epoch + 1, step, num_train_optimization_steps, tr_loss, acc, loss))
                    #     bert_model.train()

                    # define a new function to compute loss values for both output_modes
                    loss = bert_model(*batch, mode="loss")

                    if n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.

                    if self.param.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(optimizer), self.param.max_grad_norm
                        )
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            bert_model.parameters(), self.param.max_grad_norm
                        )

                    tr_loss += loss.item()
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    bert_model.zero_grad()
                    global_step += 1

                    steps.set_description(
                        "Epoch {}/{}, Batch Loss {:.7f}, Mean Loss {:.7f}".format(
                            epoch + 1, self.param.epochs, loss.item(), tr_loss / (step + 1)
                        )
                    )

                model = BertSimMatchModel(bert_model, tokenizer, config)
                acc, loss = self.evaluate(model, test_data, test_label_list)
                one_fold_acc_list.append(acc)
                logger.info(
                    "Epoch {}, train Loss: {:.7f}, eval acc: {}, eval loss: {:.7f}".format(
                        epoch + 1, tr_loss, acc, loss
                    )
                )
                bert_model.train()
            all_acc_list.append(one_fold_acc_list)
            model = BertSimMatchModel(bert_model, tokenizer, config)
            model.save(model_dir)

        logger.info("***** Stats *****")
        # 计算k-fold的平均的acc
        all_epoch_acc = list(zip(*all_acc_list))
        logger.info("acc for each epoch:")
        for epoch, acc in enumerate(all_epoch_acc, start=1):
            logger.info(
                "epoch %d, mean: %.5f, std: %.5f"
                % (epoch, float(np.mean(acc)), float(np.std(acc)))
            )

        logger.info("***** Training complete *****")

    @staticmethod
    def evaluate(model: BertSimMatchModel, data: TripletTextDataset, real_label_list: List[str]):
        """
        评估模型，计算acc

        :param model:
        :param data:
        :param real_label_list:
        :return:
        """
        num_padding = 0
        # if isinstance(model.model, torch.nn.DataParallel):
        #     num_padding = (model.predict_batch_size - len(data) % model.predict_batch_size)
        #     if num_padding != 0:
        #         padding_data = TripletTextDataset(
        #             text_a_list=[""] * num_padding,
        #             text_b_list=[""] * num_padding,
        #             text_c_list=[""] * num_padding,
        #         )
        #         data = data.__add__(padding_data)

        sampler = SequentialSampler(data)
        collate_fn = get_collator(model.max_length, model.device, model.tokenizer)
        dataloader = DataLoader(data, sampler=sampler, batch_size=8, collate_fn=collate_fn)

        predict_result = []
        loss_sum = 0
        for batch in dataloader:
            with torch.no_grad():
                output = model.model(*batch, mode="evaluate")
                loss = output[2].mean().cpu().item()
                loss_sum += loss
                predict_results = output[1].cpu().numpy()
                cata_indexes = np.argmax(predict_results, axis=1)

                for i_sample, cata_index in enumerate(cata_indexes):
                    prob = predict_results[i_sample][cata_index]
                    label = "B" if cata_index == 0 else "C"
                    predict_result.append((str(label), float(prob)))

        if num_padding != 0:
            predict_result = predict_result[:-num_padding]
        assert len(predict_result) == len(real_label_list)

        correct = 0
        for i, real_label in enumerate(real_label_list):
            try:
                predict_label = predict_result[i][0]
                if predict_label == real_label:
                    correct += 1
            except Exception as e:
                print(e)
                continue

        acc = correct / len(real_label_list)
        return acc, loss_sum
