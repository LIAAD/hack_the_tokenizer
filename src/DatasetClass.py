from torch.utils.data import Dataset

class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]

    def to_list(self): return [a for a in self.original_list]


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, batch_size=1, max_length=None):
        self.texts = texts
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Find the maximum length for the batch block
        batch_block = idx // self.batch_size
        max_length = max(len(self.tokenizer.tokenize(x)) for x in self.texts[batch_block*self.batch_size:(batch_block+1)*self.batch_size])
        inputs = self.tokenizer(
            self.texts[idx], 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_length if self.max_length is not None else max_length, 
            return_tensors='pt'
        )
        return inputs