from omegaconf import DictConfig
from regex import X
from torch import nn

from src.utils.technical_utils import load_obj
from transformers import AutoModel

class BERT(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        """
        Model class.

        Args:
            cfg: main config
        """
        super().__init__()

        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(cfg.model.params.dropout)
        # relu activation function
        self.relu = nn.ReLU()
        # dense layer 1
        self.fc1 = nn.Linear(768, 512)
        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512, cfg.model.params.n_clas)
        #softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, mask):
        _, cls_hs = self.bert(x,
                                  attention_mask=mask,
                                  return_dict=False)

        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x