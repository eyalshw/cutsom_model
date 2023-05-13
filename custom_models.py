from transformers import PreTrainedModel, AutoConfig, BertModel
import torch.nn as nn


class BertClassifier(PreTrainedModel):
    """Bert Model for Classification Tasks."""
    config_class = AutoConfig

    def __init__(self, config):  # tuning only the head

        # super(BertClassifier, self).__init__()
        super().__init__(config)
        num_labels = 20
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.config = self.bert.config

        # # Instantiate BERT model
        # # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        # self.D_in = 1024  # hidden size of Bert
        # self.H = 512
        # self.D_out = 2
        #
        # # Instantiate the classifier head with some one-layer feed-forward classifier
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.D_in, 512),
        #     nn.Tanh(),
        #     nn.Linear(512, self.D_out),
        #     nn.Tanh()
        # )
        #
        # super(CustomModel, self).__init__()
        # self.bert = BertModel.from_pretrained('bert-base-uncased')
        # self.dropout = nn.Dropout(0.1)
        # self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        # self.config = self.bert.config

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return outputs
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

        # # Feed input to BERT
        # outputs = self.bert(input_ids=input_ids,
        #                     attention_mask=attention_mask)
        #
        # # Extract the last hidden state of the token `[CLS]` for classification task
        # last_hidden_state_cls = outputs[0][:, 0, :]
        #
        # # Feed input to classifier to compute logits
        # logits = self.classifier(last_hidden_state_cls)
        #
        # return logits
