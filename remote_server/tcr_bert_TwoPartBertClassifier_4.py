import os
from typing import Literal

import torch.nn as nn
import torch
from transformers import BertModel, BertConfig, AutoModel

import fc

class TwoPartClassLogitsHead(nn.Module):
    """
    Classifier head that takes the encoded representation of TRA and TRB
    of shape (batch, hidden_dim), projects each through its own separate
    fully connected layer, before concatenating and projecting to final
    output with sigmoid activation for multi-label classification.
    """
    def __init__(self, a_enc_dim: int, b_enc_dim: int, n_out: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout, inplace=False)
        self.fc_a = fc.FullyConnectedLayer(a_enc_dim, 32, activation=nn.ReLU())
        self.fc_b = fc.FullyConnectedLayer(b_enc_dim, 32, activation=nn.ReLU())
        # Use None for the final layer activation to apply sigmoid later
        self.final_fc = fc.FullyConnectedLayer(64, n_out, activation=None)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for multi-label classification

    def forward(self, a_enc, b_enc) -> torch.Tensor:
        a_enc = self.fc_a(self.dropout(a_enc))
        b_enc = self.fc_b(self.dropout(b_enc))
        enc = torch.cat([a_enc, b_enc], dim=-1)
        logits = self.final_fc(enc)
        return self.sigmoid(logits)  # Apply sigmoid

class TwoPartBertClassifier(nn.Module):
    """
    Two part BERT model, one part each for tcr a/b, modified for multi-label classification.
    """
    def __init__(
        self,
        pretrained: str,
        n_output: int,
        freeze_encoder: bool = False,
        separate_encoders: bool = True,
        dropout: float = 0.2,
        seq_pooling: Literal["max", "mean", "cls", "pool"] = "cls",
    ):
        super().__init__()
        self.pooling_strat = seq_pooling
        self.pretrained = pretrained
        self.trans_a = self.__initialize_transformer()
        if separate_encoders:
            self.trans_b = self.__initialize_transformer()
        else:
            self.trans_b = self.trans_a

        self.trans_a.train()
        self.trans_b.train()
        if freeze_encoder:
            for param in self.trans_a.parameters():
                param.requires_grad = False
            for param in self.trans_b.parameters():
                param.requires_grad = False

        self.head = TwoPartClassLogitsHead(
            self.trans_a.encoder.layer[-1].output.dense.out_features,
            self.trans_b.encoder.layer[-1].output.dense.out_features,
            n_out=n_output,
            dropout=dropout,
        )

    def __initialize_transformer(self) -> BertModel:
        return AutoModel.from_pretrained(
            self.pretrained, add_pooling_layer=self.pooling_strat == "pool"
        )

    def forward(self, tcr_a, tcr_b):
        a_enc = self.trans_a(tcr_a)
        b_enc = self.trans_b(tcr_b)

        if self.pooling_strat == "mean":
            a_enc = a_enc.last_hidden_state.mean(dim=1)
            b_enc = b_enc.last_hidden_state.mean(dim=1)
        elif self.pooling_strat == "max":
            a_enc = a_enc.last_hidden_state.max(dim=1)[0]
            b_enc = b_enc.last_hidden_state.max(dim=1)[0]
        elif self.pooling_strat == "cls":
            a_enc = a_enc.last_hidden_state[:, 0, :]
            b_enc = b_enc.last_hidden_state[:, 0, :]
        elif self.pooling_strat == "pool":
            a_enc = a_enc.pooler_output
            b_enc = b_enc.pooler_output
        else:
            raise ValueError(f"Unrecognized pooling strategy: {self.pooling_strat}")

        return self.head(a_enc, b_enc)
