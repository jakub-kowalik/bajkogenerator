import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.generation_utils import (
    top_k_top_p_filtering,
    greedy_search,
    multinomial_sampling,
    temperature_softmax,
)


class LSTMTextGenerator(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_size,
        lstm_size,
        lstm_layers=1,
        lstm_bidirectional=False,
        lstm_dropout=0.2,
        use_gru=False,
        pad_idx=2,
        seq_len=20,
    ):
        super(LSTMTextGenerator, self).__init__()

        self.pad_idx = pad_idx

        self.seq_len = seq_len

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=emb_size, padding_idx=pad_idx
        )

        self.lstm = (
            nn.LSTM(
                input_size=emb_size,
                hidden_size=lstm_size,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=lstm_dropout,
                bidirectional=lstm_bidirectional,
            )
            if not use_gru
            else nn.GRU(
                input_size=emb_size,
                hidden_size=lstm_size,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=lstm_dropout,
                bidirectional=lstm_bidirectional,
            )
        )

        self.dropout = nn.Dropout(lstm_dropout)

        self.fc1 = nn.Linear(
            in_features=lstm_size * 2 if lstm_bidirectional else lstm_size,
            out_features=vocab_size,
        )

        # self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, inputs):  # add typing later
        embedded = self.embedding(inputs)

        lstm_output, _ = self.lstm(embedded)

        dropped = self.dropout(lstm_output[:, -1, :])
        output = self.fc1(dropped)

        return output

    def generate(
        self,
        start_text,
        length=100,
        temperature=1.0,
        strategy="top_k_top_p",
        top_k=0,
        top_p=1.0,
        n_samples=1,
    ):
        assert not self.training
        starter = start_text

        with torch.no_grad():
            for i in range(length):
                inp = torch.LongTensor([starter[-self.seq_len :]])
                pred = self.forward(inp)

                logits = pred / temperature
                if strategy == "greedy":
                    out = greedy_search(logits)
                elif strategy == "multinomial":
                    out = torch.nn.functional.softmax(logits, dim=1)
                    out = multinomial_sampling(out, n_samples=n_samples)
                elif strategy == "top_k_top_p":
                    out = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                    out = torch.nn.functional.softmax(out, dim=1)
                    out = multinomial_sampling(out, n_samples=n_samples)

                starter.append(out.item())
                if out == 1:
                    break

        return starter
