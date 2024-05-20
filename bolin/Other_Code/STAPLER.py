from transformers import PreTrainedModel, PretrainedConfig
from x_transformers.x_transformers import TransformerWrapper, Encoder
import torch.nn as nn
import torch

class STAPLERTransformerConfig(PretrainedConfig):
    model_type = "stapler_transformer"
    def __init__(self, num_tokens=25, emb_dim=25, cls_dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens
        self.emb_dim = emb_dim
        self.cls_dropout = cls_dropout
        # Add any other specific configurations here

class STAPLERTransformer(PreTrainedModel):
    config_class = STAPLERTransformerConfig
    def __init__(self, config):
        super().__init__(config)
        self.transformer = TransformerWrapper(
            num_tokens=config.num_tokens,
            attn_layers=Encoder(
                dim=config.emb_dim,
                depth=8,
                heads=8,
                ff_glu=True,
                rel_pos_bias=True,
                attn_dropout=0.4,
                ff_dropout=0.4
            )
        )
        self.to_logits = nn.Linear(config.emb_dim, config.num_tokens)
        if config.output_classification:
            self.to_cls = nn.Sequential(
                nn.Linear(config.emb_dim, config.emb_dim),
                nn.Tanh(),
                nn.Dropout(config.cls_dropout),
                nn.Linear(config.emb_dim, 2),
            )


def convert_checkpoint(original_checkpoint):
    # 这里需要根据原始的 PyTorch 模型和预期的 Hugging Face 模型架构转换权重。
    converted_weights = {}
    for key, value in original_checkpoint.items():
        # 修改 key 的名称，使其符合 Hugging Face 的命名约定
        new_key = key.replace("some_old_prefix", "new_prefix")
        converted_weights[new_key] = value
    return converted_weights



# model = STAPLERTransformer.from_pretrained("path_to_converted_checkpoint")
checkpoint = torch.load("/Users/berlin/Documents/UoB/DSMP/pycharm_project_DSMP/STAPLER/pre-cdr3_combined_epoch=437-train_mlm_loss=0.702.ckpt", map_location=torch.device('mps'))
checkpoint_2 = torch.load("/Users/berlin/Documents/UoB/DSMP/pycharm_project_DSMP/dsmp-2024-group-6/best_model_small.pth", map_location=torch.device('mps'))

print(checkpoint_2)