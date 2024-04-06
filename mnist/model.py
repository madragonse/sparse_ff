import torch
import torch.nn.functional as F
from torch import nn, optim
import pytorch_lightning as pl 
import torchmetrics 


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
        B = x.shape[0]
        ch = self.controller(x)
        ch = ch.reshape((self.sparsity*B, -1))
        ch = F.gumbel_softmax(ch, 1, True)
        c = ch.reshape((1, -1))

        h = self.l1(x)
        os = h.shape
        h = (self.act1(h).reshape((1, -1)) * c).reshape(os)
        h = self.l2(h)
        return h
    
    def forward_gen(self, x):
        B = x.shape[0]
        ch = self.controller(x)
        ch = ch.reshape((self.sparsity*B, -1))

        ix = torch.argmax(ch, dim=-1).unsqueeze(1)
        zch = torch.zeros_like(ch)
        ch = zch.scatter(1, ix, 1)
        c = ch.reshape((1, -1))

        h = self.l1(x)
        os = h.shape
        h = (self.act1(h).reshape((1, -1)) * c).reshape(os)
        h = self.l2(h)
        return h

    def forward_inf(self, x):
        sample = x
        ch = self.controller(sample)
        ch = ch.reshape((self.sparsity, -1))
        c = torch.argmax(ch, dim=-1)

        s1, s2 = self.l1.weight.shape
        rw = self.l1.weight.reshape(self.sparsity, -1, s2)
        rb = self.l1.bias.reshape(self.sparsity, -1)
        h = sample @ rw[torch.arange(rw.size(0)), c].reshape(-1, s2).T\
            + rb[torch.arange(rb.size(0)), c].reshape(-1)

        h = self.act1(h)

        w2 = self.l2.weight.T
        s1, s2 = w2.shape
        rw = w2.reshape(self.sparsity, -1, s2)
        h = h @ rw[torch.arange(rw.size(0)), c].reshape(-1, s2)\
            + self.l2.bias

        return h

    
class FF(nn.Module):
    def __init__(self, d_model, d_hidden=None):
        super().__init__()
        d_hidden = d_model*4 if d_hidden is None else d_hidden

        self.l1 = nn.Linear(d_model, d_hidden)
        self.act1 = nn.ReLU()
        self.l2 = nn.Linear(d_hidden, d_model)

    def forward(self, x):
        h = self.l1(x)
        h = self.act1(h)
        h = self.l2(h)
        return h


class SparseNN(pl.LightningModule):
    def __init__(self, input_size, num_classes, learning_rate, hidden_size=50, low_rank=2, sparsity=1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.lr = learning_rate
        self.sparsity = sparsity
        
        self.l1 = nn.Linear(input_size, hidden_size)
        if sparsity:
            self.ff = SparseFF(hidden_size, hidden_size*4, low_rank, sparsity)
        else:
            self.ff = FF(hidden_size, hidden_size*4)
            self.ff.forward_gen = self.ff.forward
            self.ff.forward_inf = self.ff.forward
        self.l2 = nn.Linear(hidden_size, num_classes)

        self.loss = nn.CrossEntropyLoss()
        
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, inputs):
        h = self.l1(inputs)
        h = self.ff(h)
        h = self.l2(h)
        return h

    def forward_gen(self, inputs):
        h = self.l1(inputs)
        h = self.ff.forward_gen(h)
        h = self.l2(h)
        return h
    
    def forward_inf(self, inputs):
        h = self.l1(inputs)
        h = self.ff.forward_inf(h)
        h = self.l2(h)
        return h
    
    def _common_step(self, batch, batch_idx):
        X, Y = batch
        X = X.reshape(X.shape[0], -1)
        pred = self.forward(X)
        loss = self.loss(pred, Y)
        return loss, pred, Y
    
    def _common_step_gen(self, batch, batch_idx):
        X, Y = batch
        X = X.reshape(X.shape[0], -1)
        pred = self.forward_gen(X)
        loss = self.loss(pred, Y)
        return loss, pred, Y        
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, Y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, Y)

        self.log_dict({
            'train_loss': loss, 
            'train_acc': accuracy, 
        }, on_step=False, on_epoch=True, prog_bar=True)

        return {'loss': loss, "scores": scores, "Y": Y}
    
    def validation_step(self, batch, batch_idx):
        loss, pred, Y = self._common_step_gen(batch, batch_idx)
        acc = self.accuracy(pred, Y)
        self.log_dict({'val_loss': loss, 'val_acc': acc})
        return loss

    def test_step(self, batch, batch_idx):
        loss, pred, Y = self._common_step_gen(batch, batch_idx)
        acc = self.accuracy(pred, Y)
        self.log_dict({'test_loss': loss, 'test_acc': acc})
        return loss

    def predict_step(self, x):
        x = x.reshape([1, -1])
        scores = self.forward_inf(x)
        preds = torch.argmax(scores, dim=1)
        return preds
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
