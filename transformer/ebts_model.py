# model module

import torch
import torch.nn.functional as F
from torch import nn, optim
import pytorch_lightning as pl 


class SparseFF(nn.Module):
    def __init__(self, d_model, d_hidden=None, low_rank=2, sparsity=1):
        super().__init__()
        self.d_model = d_model
        self.sparsity = sparsity
        d_hidden = d_model*4 if d_hidden is None else d_hidden
        
        assert d_hidden%sparsity == 0
        assert low_rank >= sparsity

        self.l1 = nn.Linear(d_model, d_hidden)
        self.act1 = nn.ReLU()
        self.l2 = nn.Linear(d_hidden, d_model)

        self.controller = nn.Sequential(
            nn.Linear(d_model, low_rank),
            nn.ReLU(), # !!!
            nn.Linear(low_rank, d_hidden)
        )

    def forward(self, x):
        B, T, C = x.shape
        ch = self.controller(x)
        ch = ch.reshape((self.sparsity*B*T, -1))
        ch = F.gumbel_softmax(ch, 1, True)
        c = ch.reshape((1, -1))

        h = self.l1(x)
        os = h.shape
        h = (self.act1(h).reshape((1, -1)) * c).reshape(os)
        h = self.l2(h)
        h = h.reshape(B, T, -1)
        return h
    
    def forward_gen(self, x):
        B, T, C = x.shape
        ch = self.controller(x)
        ch = ch.reshape((self.sparsity*B*T, -1))
        ix = torch.argmax(ch, dim=-1).unsqueeze(1)
        zch = torch.zeros_like(ch)
        ch = zch.scatter(1, ix, 1)
        c = ch.reshape((1, -1))

        h = self.l1(x)
        os = h.shape
        h = (self.act1(h).reshape((1, -1)) * c).reshape(os)
        h = self.l2(h)
        h = h.reshape(B, T, -1)
        return h

    def forward_inf(self, x):
        B, T, C = x.shape
        sample = x[0]
        ch = self.controller(sample)
        ch = ch.reshape((self.sparsity*B*T, -1))
        c = torch.argmax(ch, dim=-1)

        w1 = self.l1.weight
        s1, s2 = w1.shape
        rw = w1.reshape(self.sparsity, -1, s2)
        rb = self.l1.bias.reshape(self.sparsity, -1)
        a = torch.arange(self.sparsity)
        x_range = torch.cat([a for _ in range(T)])
        h = sample @ rw[x_range, c].reshape(-1, s2).T\
            + rb[x_range, c].reshape(-1)

        h = self.act1(h)

        w2 = self.l2.weight.T
        s1, s2 = w2.shape
        rw = w2.reshape(self.sparsity, -1, s2)
        h = h @ rw[x_range, c].reshape(-1, s2)\
            + self.l2.bias

        return h.reshape(B, T, -1)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class EncoderBlock(nn.Module):

    def __init__(self, n_embd, n_head, block_size, dropout, low_rank, sparsity):
        super().__init__()
        self.block_size = block_size
        self.sa = nn.MultiheadAttention(n_embd, n_head, dropout, batch_first=True)
        self.ffwd = SparseFF(n_embd, n_embd*4, low_rank, sparsity)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.register_buffer('attn_mask', generate_square_subsequent_mask(self.block_size))
    
    def forward(self, x):
        if issubclass(type(x), tuple):
            x = x[0]
            h = self.ln1(x)
            attn_output, attn_output_weights = self.sa.forward(h, h, h, attn_mask=self.attn_mask)
            x = x + attn_output
            x = x + self.ffwd.forward_gen(self.ln2(x))
            return (x,)
        elif issubclass(type(x), list):
            x = x[0]
            h = self.ln1(x)
            attn_output, attn_output_weights = self.sa.forward(h, h, h, attn_mask=self.attn_mask)
            x = x + attn_output
            x = x + self.ffwd.forward_inf(self.ln2(x))
            return [x,]
        else:
            h = self.ln1(x)
            attn_output, attn_output_weights = self.sa.forward(h, h, h, attn_mask=self.attn_mask)
            x = x + attn_output
            x = x + self.ffwd(self.ln2(x))
            return x
    

class EBSTransformerModel(pl.LightningModule):
    def __init__(self, vocab_size, n_embd, block_size, n_head, dropout, n_layer, learning_rate, sparsity, low_rank) -> None:
        super().__init__()

        self.lr = learning_rate
        self.sparsity = sparsity
        self.low_rank = low_rank

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[EncoderBlock(n_embd, n_head, block_size, dropout, self.low_rank, self.sparsity) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        self.block_size = block_size
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, inputs):
        B, T = inputs.shape
        tok_emb = self.token_embedding_table(inputs) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        return logits
    
    def forward_gen(self, inputs):
        B, T = inputs.shape
        tok_emb = self.token_embedding_table(inputs)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb
        x = self.blocks((x,))[0]
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
    
    def forward_inf(self, inputs):
        B, T = inputs.shape
        tok_emb = self.token_embedding_table(inputs)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb
        x = self.blocks([x,])[0]
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
    
    def _common_step(self, batch, batch_idx):
        X, targets = batch
        logits = self.forward(X)

        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = self.loss(logits, targets)
        return loss, logits, targets
    
    def _validation_step(self, batch, batch_idx):
        X, targets = batch
        logits = self.forward_gen(X)

        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = self.loss(logits, targets)
        return loss, logits, targets
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, Y = self._common_step(batch, batch_idx)

        self.log_dict({
            'train_loss': loss, 
        }, on_step=False, on_epoch=True, prog_bar=True)

        return {'loss': loss, "scores": scores, "Y": Y}
    
    def validation_step(self, batch, batch_idx):
        loss, pred, Y = self._validation_step(batch, batch_idx)
        self.log_dict({'val_loss': loss})
        return loss

    def test_step(self, batch, batch_idx):
        loss, pred, Y = self._common_step(batch, batch_idx)
        self.log_dict({'test_loss': loss})
        return loss

    def predict_step(self, x):
        scores = self.forward_inf(x)
        preds = torch.argmax(scores, dim=2)
        return preds
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
