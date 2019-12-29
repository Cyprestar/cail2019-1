import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from transformers.modeling_bert import BertPreTrainedModel, BertModel


class BertForSimMatchModel(BertPreTrainedModel):
    """
    ab、ac交互并编码
    """

    def __init__(self, config):
        super(BertForSimMatchModel, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)
        self.init_weights()

    def forward(self, ab, ac, labels=None, mode="prob"):
        ab_pooled_output = self.bert(*ab)[1]
        ac_pooled_output = self.bert(*ac)[1]
        subtraction_output = ab_pooled_output - ac_pooled_output
        concated_pooled_output = self.dropout(subtraction_output)
        output = self.seq_relationship(concated_pooled_output)

        if mode == "prob":
            prob = torch.nn.functional.softmax(Variable(output), dim=1)
            return prob
        elif mode == "logits":
            return output
        elif mode == "loss":
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(output.view(-1, 2), labels.view(-1))
            return loss
        elif mode == "evaluate":
            prob = torch.nn.functional.softmax(Variable(output), dim=1)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(output.view(-1, 2), labels.view(-1))
            return output, prob, loss
