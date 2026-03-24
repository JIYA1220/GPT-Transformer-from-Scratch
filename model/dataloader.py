import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


class GPTDatasetV1(Dataset):
    """Memory-efficient sliding-window tokenized text dataset."""

    def __init__(self, txt, tokenizer, max_length, stride):
        # Store as one giant tensor instead of many small ones
        self.token_ids = torch.tensor(
            tokenizer.encode(txt, allowed_special={"<|endoftext|>"}),
            dtype=torch.long
        )
        self.max_length = max_length
        self.stride     = stride

        # Pre-calculate indices to avoid overhead in __getitem__
        self.indices = list(range(0, len(self.token_ids) - max_length, stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        input_chunk  = self.token_ids[start_idx : start_idx + self.max_length]
        target_chunk = self.token_ids[start_idx + 1 : start_idx + self.max_length + 1]
        return input_chunk, target_chunk


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    """Create a DataLoader from raw text using GPT-2 tokenizer."""
    tokenizer  = tiktoken.get_encoding("gpt2")
    dataset    = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    return dataloader