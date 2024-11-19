


def get_byte_pairs_frequency(byte_list: bytes) -> dict[tuple[int, int], int]:
    '''
        This function takes a list of bytes and returns a dictionary of byte pairs and their frequencies.
    '''
    byte_pairs = {}
    for byte1, byte2 in zip(byte_list[:-1], byte_list[1:]):
        byte_pair = (byte1, byte2)
        byte_pairs.setdefault(byte_pair, 0)

        byte_pairs[byte_pair] += 1
    return byte_pairs


def byte_pair_encoding(input_text: str, vocab_size: int):
    '''
        This function takes a tokenizer and a text and returns a list of tokens.
    '''
    # Convert input text to bytes
    text = input_text.encode('utf-8')

    # Output variable
    vocab = {i: bytes([i]) for i in text}
    cur_token_index = max(vocab.items(), key=lambda x: x[0])[0] + 1    # Which "id" to assign to the next token
    initial_vocab_size = len(vocab)

    # Iterate until `vocab_size` is reached
    while len(vocab) < vocab_size + initial_vocab_size:
        # Obtain the frequency of all byte pairs
        byte_pairs_frequency = get_byte_pairs_frequency(text)
        
        # Select the most frequent byte pair
        selected_byte_pair = max(byte_pairs_frequency.items(), key=lambda x: x[1])[0]

        # Add it to our vocabulary
        vocab[cur_token_index] = bytes(selected_byte_pair)

        # Replace the most frequent bytes with the new byte pair
        text = text.replace(vocab[cur_token_index], bytes([cur_token_index+1]))
        cur_token_index += 1
    return vocab




if __name__ == '__main__':
    print(byte_pair_encoding('Hello, my name is michale and what is yours?', 3))