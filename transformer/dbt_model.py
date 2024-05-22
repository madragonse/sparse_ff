# model module

from typing import Any
import torch
from torch import nn, optim
from torch.utils.data import random_split
import pytorch_lightning as pl 

from utils import generate_square_subsequent_mask

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    

class DecoderBlock(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, block_size, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.block_size = block_size
        self.sa = nn.MultiheadAttention(n_embd, n_head, dropout, batch_first=True)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.register_buffer('attn_mask', generate_square_subsequent_mask(self.block_size))
    
    def forward(self, x):
        h = self.ln1(x)
        attn_output, attn_output_weights = self.sa.forward(h, h, h, attn_mask=self.attn_mask) # 
        x = x + attn_output
        x = x + self.ffwd(self.ln2(x))
        return x
    
class DBTransformerModel(pl.LightningModule):
    def __init__(self, vocab_size, n_embd, block_size, n_head, dropout, n_layer, learning_rate, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.lr = learning_rate

        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[DecoderBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.block_size = block_size
        self.loss = nn.CrossEntropyLoss() #nn.TransformerEncoderLayer
    
    def forward(self, inputs):
        # set breakpoint
        # from pdb import set_trace; set_trace()
        
        B, T = inputs.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(inputs) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        return logits
    
    def _common_step(self, batch, batch_idx):
        X, targets = batch
        logits = self.forward(X)

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
        loss, pred, Y = self._common_step(batch, batch_idx)
        self.log_dict({'val_loss': loss})
        return loss

    def test_step(self, batch, batch_idx):
        loss, pred, Y = self._common_step(batch, batch_idx)
        self.log_dict({'test_loss': loss})
        return loss

    def predict_step(self, x):
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
