from omegaconf import DictConfig
from regex import X
from torch import nn

from methazAI.utils.technical_utils import load_obj
from transformers import AutoModelForSequenceClassification 

class LilYuanModel(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        """
        Model class.

        Args:
            cfg: main config
        """
        super().__init__()

        self.model = AutoModelForSequenceClassification.from_pretrained(
            "LiYuan/amazon-review-sentiment-analysis",
            num_labels=cfg.model.params.n_clas,
            return_dict=cfg.model.params.return_dict,
        )
        print(self.model)
        if cfg.model.params.finetune:
            for param in self.model.bert.parameters():
                param.requires_grad = False

    def forward(self, x, mask):
        output = self.model(x, mask)
        return output[0]