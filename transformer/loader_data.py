
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class ShakespeareDataset(Dataset):
    def __init__(self, data_dir, block_size, train=True) -> None:
        with open(data_dir, 'r', encoding='utf-8') as f:
            text = f.read()
        self.block_size = block_size

        # here are all the unique characters that occur in this text
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        # create a mapping from characters to integers
        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = { i:ch for i,ch in enumerate(chars) }
        self.encoder = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
        self.decoder = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

        entire_dataset = torch.tensor(self.encoder(text), dtype=torch.long)

        n = int(0.9*len(entire_dataset)) # first 90% will be train, rest val

        self.train = train
        if self.train:
            self.dataset = entire_dataset[:n]
        else:
            self.dataset = entire_dataset[n:]

    def __len__(self):
        return len(self.dataset)-self.block_size-1

    def __getitem__(self, idx):
        x = self.dataset[idx:idx+self.block_size]
        y = self.dataset[idx+1:idx+self.block_size+1]
        return x, y


class ShakespeareDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, block_size, num_workers=0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.block_size = block_size

        with open(self.data_dir, 'r', encoding='utf-8') as f:
            text = f.read()
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)

    def prepare_data(self):
        pass

    def setup(self, stage):
        self.train_ds = ShakespeareDataset(self.data_dir, self.block_size, train=True)
        self.val_ds = ShakespeareDataset(self.data_dir, self.block_size, train=False)
        

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

