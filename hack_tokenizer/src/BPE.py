# install and import libraries
from collections import Counter, defaultdict
from transformers import AutoTokenizer
import tqdm

class BPE():
    """Byte-Pair Encoding: Subword-based tokenization algorithm."""
    
    def __init__(self, corpus, vocab_size):
        """Initialize BPE tokenizer."""
        self.corpus = corpus
        self.vocab_size = vocab_size
        
        # pre-tokenize the corpus into words, BERT pre-tokenizer is used here
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.word_freqs = defaultdict(int)
        self.splits = {}
        self.merges = {}
        self.vocab = []
        self.vocab_range = 255  # Variable used to assign new token_ids (meaning they will start being assigned with vocab_range + 1)
        self.dictionary = {}

    def update_vocab(self, new_token, pos=None):
        if isinstance(new_token, bytes):
            new_token = tuple(new_token)
        self.vocab_range += 1
        if pos is not None:
            self.vocab.insert(pos, self.vocab_range)
        else: 
            self.vocab.append(self.vocab_range)

        # After updating "vocabulary", update "dictionary"
        self.dictionary[self.vocab_range] = new_token
        self.dictionary[new_token] = self.vocab_range
        return self.vocab
    
    def train(self):
        """Train BPE tokenizer."""

        # compute the frequencies of each word in the corpus
        for text in tqdm.tqdm(self.corpus, desc='<BPE.train> Computing word frequencies'):
            words_with_offsets = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
            new_words = [word.encode('utf-8') for word, offset in words_with_offsets] # Words can finish with a "space" by the end
            for word in new_words:
                self.word_freqs[word] += 1

        # compute the base vocabulary of all characters in the corpus
        for word in self.word_freqs.keys():
            for letter in word:
                if letter not in self.vocab:
                    self.vocab.append(letter)
                    self.dictionary[letter] = tuple(bytes([letter]))
                    self.dictionary[tuple(bytes([letter]))] = letter
        self.vocab.sort()

        # add the special token </w> [with a token id of 256 (bytes are only up to 255)] at the beginning of the vocabulary (and to our dictionary)
        self.update_vocab('</w>'.encode('utf-8'), pos=0)

        # split each word into individual characters before training
        self.splits = {word: [c for c in word] for word in self.word_freqs.keys()}

        # merge the most frequent pair iteratively until the vocabulary size is reached
        for _ in tqdm.tqdm(range(self.vocab_size - len(self.vocab)), desc='<BPE.train> Computing Merges'):

            # compute the frequency of each pair
            pair_freqs = self.compute_pair_freqs()

            # find the most frequent pair
            best_pair = max(pair_freqs.items(), key=lambda x: x[1])[0]

            # Update vocabulary to include it
            self.update_vocab(best_pair)

            # merge the most frequent pair
            self.merge_pair(*best_pair)
            self.merges[best_pair] = self.dictionary[best_pair]


    def compute_pair_freqs(self):
        """Compute the frequency of each pair."""

        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = self.splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs


    def merge_pair(self, a, b):
        """Merge the given pair."""

        for word in self.word_freqs:
            split = self.splits[word]
            if len(split) == 1:
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [self.dictionary[(a, b)]] + split[i + 2 :]
                else:
                    i += 1
            self.splits[word] = split
        return self.splits
    

    def tokenize(self, text: str):
        """Tokenize a given text with trained BPE tokenizer (including pre-tokenization, split, and merge)."""

        splits_text = [char for char in text.encode('utf-8')]

        for pair, merge in self.merges.items():
            new_split_text = []
            skip_next = False
            for t1, t2 in zip(splits_text[:-1], splits_text[1:]):
                if skip_next:
                    skip_next = False
                    continue
                if pair == (t1, t2):
                    new_split_text.append(merge)
                    skip_next = True
                    continue
                new_split_text.append(t1)
            # Case where don't skip the last token (means we haven't added it yet)
            if not skip_next:
                new_split_text.append(splits_text[-1])
            splits_text = new_split_text
        return splits_text

    def token_from_id(self, token_id: int):
        if token_id < 256: return bytes([token_id])
        return b''.join([self.token_from_id(x) for x in self.dictionary[token_id]])

    def from_id_to_tokens(self, tokens: list[int], byte_decode: bool=False):
        decode_func = lambda x: x if not byte_decode else x.decode('utf-8')
        return [decode_func(self.token_from_id(token)) for token in tokens]

    def decode(self, vector: list[int]):
        out_vector = vector.copy()
        
        while max(out_vector) > 255:
            new_vec = []
            for token in out_vector:
                new_vec.extend(self.dictionary.get(token, (token, )))
            out_vector = new_vec
        return bytes(new_vec).decode('utf-8')
