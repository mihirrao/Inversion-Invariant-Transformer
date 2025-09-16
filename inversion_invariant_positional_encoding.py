import torch
import torch.nn as nn

class InversionInvariantPositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

        # inversion invariance
        pivot = max_len/2
        position = torch.tensor([x if x < pivot else max_len - x - 1 for x in torch.arange(max_len)]).unsqueeze(1)
        print(position)

        div_term = torch.exp(torch.arange(0, dim_model, 2) * (-torch.log(torch.tensor(10000.0)) / dim_model))
        pe = torch.zeros(1, max_len, dim_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, num_tokens, dim_model, num_heads, num_encoder_layers, num_decoder_layers, dropout_p):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.positional_encoder = InversionInvariantPositionalEncoding(dim_model=dim_model, dropout_p=dropout_p, max_len=5000)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
            batch_first=True
        )
        self.out = nn.Linear(dim_model, num_tokens)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        src_embedded = self.positional_encoder(self.embedding(src))
        tgt_embedded = self.positional_encoder(self.embedding(tgt))

        output = self.transformer(
            src_embedded, tgt_embedded,
            src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return self.out(output)

model = Transformer(num_tokens=10000, dim_model=512, num_heads=8, num_encoder_layers=6, num_decoder_layers=6, dropout_p=0.1)
model.forward(torch.randint(0, 10000, (32, 100)), torch.randint(0, 10000, (32, 100)))