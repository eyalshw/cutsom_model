from transformers import PreTrainedModel, AutoConfig
import torch.nn as nn


class BertClassifier(PreTrainedModel):
    """Bert Model for Classification Tasks."""
    config_class = AutoConfig

    def __init__(self, config, freeze_bert=True):  # tuning only the head
        """
         @param    bert: a BertModel object
         @param    classifier: a torch.nn.Module classifier
         @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        # super(BertClassifier, self).__init__()
        super().__init__(config)

        # Instantiate BERT model
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        self.D_in = 1024  # hidden size of Bert
        self.H = 512
        self.D_out = 2

        # Instantiate the classifier head with some one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.D_in, 512),
            nn.Tanh(),
            nn.Linear(512, self.D_out),
            nn.Tanh()
        )

    def forward(self, input_ids, attention_mask):
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits
