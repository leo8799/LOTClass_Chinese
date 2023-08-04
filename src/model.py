import sys
from transformers import BertPreTrainedModel, BertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from torch import nn
from config.configs_interface import configs as args


class LOTClassModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel.from_pretrained(args.train_args.pretrained_weights_path)
        self.cls = BertOnlyMLMHead(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

        # MLM head is not trained
        for param in self.cls.parameters():
            param.requires_grad = False

    def forward(self, input_ids, pred_mode, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None):
        bert_outputs = self.bert(input_ids,  # 句子每个分词对应词典的id
                                 attention_mask=attention_mask,  # 指示mask哪些分词,被mask的分词其下标对应的值设为1
                                 token_type_ids=token_type_ids,  # 输入的是两个句子，第一个句子的所有分词对应的下标的值为0，第二个句子为1
                                 position_ids=position_ids,  # 指示每个分词在句子中的位置，具体的值是位置编码
                                 head_mask=head_mask,  # mask无效的注意力头
                                 inputs_embeds=inputs_embeds)  # 可以取代input_ids参数，直接将inputs_embeds嵌入向量输入
        last_hidden_states = bert_outputs[0]  # 0:last_hidden_state 1:pooler_output 2:hidden_states 3:attentions
        if pred_mode == "classification":
            trans_states = self.dense(last_hidden_states)
            trans_states = self.activation(trans_states)
            trans_states = self.dropout(trans_states)
            logits = self.classifier(trans_states)
        elif pred_mode == "mlm":
            logits = self.cls(last_hidden_states)
        else:
            sys.exit("Wrong pred_mode!")
        return logits

