import transformers

NEW_TOKEN = 'martelo'
NEW_TOKEN_ID = 49151

# Load the model and tokenizer
model_name = "HuggingFaceTB/SmolLM-135M" 
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)



# Delete previous token with the "new_token_id"
old_token = tokenizer.convert_ids_to_tokens(NEW_TOKEN_ID)   # Finding old token
if old_token in tokenizer.vocab:
    tokenizer.vocab.pop(old_token) # Deleting old token from vocab
if old_token in tokenizer.added_tokens_encoder:
    tokenizer.added_tokens_encoder.pop(old_token)   # And from added tokens if it's in there

# Setting up the ID to the new Token
tokenizer.vocab[NEW_TOKEN] = NEW_TOKEN_ID
tokenizer.added_tokens_encoder[NEW_TOKEN] = NEW_TOKEN_ID
tokenizer.added_tokens_decoder[NEW_TOKEN_ID] = NEW_TOKEN


print(f'{tokenizer.encode(NEW_TOKEN) = }')      # = [39339, 35304]
print(f'{tokenizer.decode(NEW_TOKEN_ID) = }')   # = 'ectable'


# -----------------------------------------------
# Trying to use the edited Tokenizer file
tokenizer = transformers.AutoTokenizer.from_pretrained('/home/yakim/hack_the_tokenizer/testing')
print(f'{tokenizer.encode(NEW_TOKEN) = }')      # = [39339, 35304]
print(f'{tokenizer.decode(NEW_TOKEN_ID) = }')   # = 'ectable'