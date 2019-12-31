import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from transformers.modeling_bert import BertPreTrainedModel, BertModel

from .esim.layers import RNNDropout, Seq2SeqEncoder, SoftmaxAttention
from .esim.utils import replace_masked


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

        self._rnn_dropout = RNNDropout(p=config.hidden_dropout_prob)
        self._attention = SoftmaxAttention()
        self._projection = nn.Sequential(nn.Linear(4 * config.hidden_size, config.hidden_size),
                                         nn.ReLU())
        self._composition = Seq2SeqEncoder(nn.LSTM,
                                           config.hidden_size,
                                           config.hidden_size,
                                           bidirectional=True)
        self._classification = nn.Sequential(nn.Dropout(p=config.hidden_dropout_prob),
                                             nn.Linear(4 * 2 * config.hidden_size, config.hidden_size),
                                             nn.Tanh(),
                                             nn.Dropout(p=config.hidden_dropout_prob),
                                             nn.Linear(config.hidden_size, 2))
        self.apply(self.init_esim_weights)

    def forward(self, ab, ac, labels=None, mode="prob"):
        ab_mask = ab[1].float()
        ac_mask = ac[1].float()
        ab_length = ab_mask.sum(dim=-1).long()
        ac_length = ac_mask.sum(dim=-1).long()
        # the parameter is: input_ids, attention_mask, token_type_ids
        # which is corresponding to input_ids, input_mask and segment_ids in InputFeatures
        ab_pooled_output = self.bert(*ab)[0]
        ac_pooled_output = self.bert(*ac)[0]
        # ab_pooled_output = self.bert(*ab)[1].flatten()
        # ac_pooled_output = self.bert(*ac)[1].flatten()
        # The return value: sequence_output, pooled_output, (hidden_states), (attentions)

        attended_ab, attended_ac = self._attention(ab_pooled_output, ab_mask, ac_pooled_output, ac_mask)

        enhanced_ab = torch.cat([ab_pooled_output,
                                 attended_ab,
                                 ab_pooled_output - attended_ab,
                                 ab_pooled_output * attended_ab],
                                dim=-1)

        enhanced_ac = torch.cat([ac_pooled_output,
                                 attended_ac,
                                 ac_pooled_output - attended_ac,
                                 ac_pooled_output * attended_ac],
                                dim=-1)

        projected_ab = self._projection(enhanced_ab)
        projected_ac = self._projection(enhanced_ac)

        # projected_ab = self._rnn_dropout(projected_ab)
        # projected_ac = self._rnn_dropout(projected_ac)

        v_ai = self._composition(projected_ab, ab_length)
        v_bj = self._composition(projected_ac, ac_length)

        v_a_avg = torch.sum(v_ai * ab_mask.unsqueeze(1)
                            .transpose(2, 1), dim=1) / torch.sum(ab_mask, dim=1, keepdim=True)
        v_b_avg = torch.sum(v_bj * ac_mask.unsqueeze(1)
                            .transpose(2, 1), dim=1) / torch.sum(ac_mask, dim=1, keepdim=True)

        v_a_max, _ = replace_masked(v_ai, ab_mask, -1e7).max(dim=1)
        v_b_max, _ = replace_masked(v_bj, ac_mask, -1e7).max(dim=1)

        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        output = self._classification(v)

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

    @staticmethod
    def init_esim_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            nn.init.constant_(module.bias.data, 0.0)
        elif isinstance(module, nn.LSTM):
            nn.init.xavier_uniform_(module.weight_ih_l0.data)
            nn.init.orthogonal_(module.weight_hh_l0.data)
            nn.init.constant_(module.bias_ih_l0.data, 0.0)
            nn.init.constant_(module.bias_hh_l0.data, 0.0)
            hidden_size = module.bias_hh_l0.data.shape[0] // 4
            module.bias_hh_l0.data[hidden_size:(2 * hidden_size)] = 1.0
            if module.bidirectional:
                nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
                nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
                nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
                nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
                module.bias_hh_l0_reverse.data[hidden_size:(2 * hidden_size)] = 1.0
