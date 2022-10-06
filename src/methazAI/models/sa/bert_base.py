from omegaconf import DictConfig
from regex import X
from torch import nn

from methazAI.utils.technical_utils import load_obj
from transformers import BertForSequenceClassification


class BERT(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        """
        Model class.

        Args:
            cfg: main config
        """
        super().__init__()

        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = cfg.model.params.n_clas, return_dict = cfg.model.params.return_dict)

        #self.model.classifier.add_module('bert_activation', nn.ReLU())
        #self.model.classifier.add_module('prediction', nn.Linear(cfg.model.params.hidden_size, cfg.model.params.n_clas))


        if cfg.model.params.finetune:
            for param in self.model.bert.parameters():
                param.requires_grad = False

            for param in self.model.classifier.parameters():
                param.requires_grad = True

    def forward(self, x, mask):
        output = self.model(x, mask)
        return output['logits']
