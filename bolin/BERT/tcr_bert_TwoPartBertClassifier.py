import os
from typing import Literal

import torch.nn as nn
# from safetensors import torch
import torch
from transformers import BertModel, BertConfig, AutoModel
# import featurization as ft
import fc


class TwoPartClassLogitsHead(nn.Module):
    """
    Classifier head that takes the encoded representation of TRA and TRB
    of shape (batch, hidden_dim), projects each through its own separate
    fully connected layer, before concatenating and projecting to final
    output
    """

    def __init__(
        self, a_enc_dim: int, b_enc_dim: int, n_out: int = 2, dropout: float = 0.1
    ):
        # BertForSequenceClassification adds, on top of the pooler (dense, tanh),
        # dropout layer with p=0.1
        # classifier linear layer
        super().__init__()
        self.dropout = nn.Dropout(p=dropout, inplace=False)
        self.fc_a = fc.FullyConnectedLayer(a_enc_dim, 32, activation=nn.ReLU())
        self.fc_b = fc.FullyConnectedLayer(b_enc_dim, 32, activation=nn.ReLU())
        self.final_fc = fc.FullyConnectedLayer(64, n_out, None)

    def forward(self, a_enc, b_enc) -> torch.Tensor:
        a_enc = self.fc_a(self.dropout(a_enc))
        b_enc = self.fc_b(self.dropout(b_enc))
        enc = torch.cat([a_enc, b_enc], dim=-1)
        retval = self.final_fc(enc)
        return retval

class TwoPartBertClassifier(nn.Module):
    """
    Two part BERT model, one part each for tcr a/b
    """

    def __init__(
        self,
        pretrained: str,
        n_output: int = 2,
        freeze_encoder: bool = False,
        separate_encoders: bool = True,
        dropout: float = 0.2,
        seq_pooling: Literal["max", "mean", "cls", "pool"] = "cls",
    ):
        super().__init__()
        self.pooling_strat = seq_pooling
        self.pretrained = pretrained
        # BertPooler takes the hidden state corresponding to the first token
        # and applies:
        # - linear from hidden_size -> hidden_size
        # - Tanh activation elementwise
        # This is somewhat misleading since it really only looks at first token
        # while the "pooling" name implies looking at the whole sequence
        # See BertPooler class in https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py
        # Do not add pooler to model unless we are using the pool
        self.trans_a = self.__initialize_transformer()
        if separate_encoders:
            self.trans_b = self.__initialize_transformer()
        else:
            self.trans_b = self.trans_a
        self.bert_variant = self.trans_a.base_model_prefix

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
        """
        Centralized helper function to initialize transformer for
        encoding input sequences
        """
        # if os.path.isfile(self.pretrained) and self.pretrained.endswith(".json"):
        #
        #     params = utils.load_json_params(self.pretrained)
        #     config = BertConfig(
        #         **params,
        #         vocab_size=len(ft.AMINO_ACIDS_WITH_ALL_ADDITIONAL),
        #         pad_token_id=ft.AMINO_ACIDS_WITH_ALL_ADDITIONAL_TO_IDX[ft.PAD],
        #     )
        #     retval = BertModel(config)
        # else:
        retval = AutoModel.from_pretrained(
            self.pretrained, add_pooling_layer=self.pooling_strat == "pool"
        )
        return retval

    def forward(self, tcr_a, tcr_b):
        bs = tcr_a.shape[0]
        # input of (batch, seq_len) (batch, 20)
        # last_hidden_state (batch, seq_len, hidden_size) (batch, 20, 144)
        # pooler_output (batch_size, hiddensize)
        if self.bert_variant == "bert":
            a_enc = self.trans_a(tcr_a)
            b_enc = self.trans_a(tcr_b)
        else:
            # a_enc = self.trans_a(tcr_a)[0]
            # b_enc = self.trans_b(tcr_b)[0]
            raise NotImplementedError

        # Perform seq pooling
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

        retval = self.head(a_enc, b_enc)
        return retval