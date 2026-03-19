import re
from importlib.metadata import version
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

print("tiktoken version: " + version("tiktoken"))

with open("/Users/rino/LLMs_From_Scratch/LLMs_from_scratch/the-verdict.txt", "r", encoding="utf-8") as f: #Imports the text file into the python file
    raw_text = f.read()
print("Total Number of characters: ", len(raw_text)) #reads the total number of characters in the text
print(raw_text[:99]) # Prints the first hundred characters of the text

#Test Code to see how the split code words
"""test_data = "Hello, Word. This is-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', test_data)
#print(result)
result = [item for item in result if item.strip()]
print(result)
"""

preprocessed_data = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed_data = [item for item in preprocessed_data if item.strip()]
print("Preprocessed data length: ", len(preprocessed_data))
print(preprocessed_data[:30])

#Converting tokens into token IDs
all_words = sorted(set(preprocessed_data))
vocab_size = len(all_words)
print(vocab_size)

vocab = {token:integer for integer, token in enumerate(all_words)}
"""for i, item in enumerate(vocab.items()):
    print(item)
    if i>= 50:
        break"""

""" Simple Tokenizer Version 1
class SimpleTokenizerV1: 
    def __init__(self, vocab):
        self.str_to_int = vocab #Just uses the same format as integer and string in the dict
        self.int_to_str = {i:s for s, i in vocab.items()} #Reverse of the original dictionary

    def encode(self, text):
        preprocessed_data = re.split(r'([,.:;?_!"()\']|--|\s)', text) #Cleanly strips the input text
        preprocessed_data = [item.strip() for item in preprocessed_data if item.strip()] #Removes any unnecessary white spaces
        ids = [self.str_to_int[s] for s in preprocessed_data] #This uses orginial dict to get the strings from the words in the stripped text input
        return ids
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids]) #Joins all the words after getting them from the reverse dict with spaces in between

        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) #Removes spaces in front of all the punctuation characters
        return text
    
tokenizer = SimpleTokenizerV1(vocab) #Tokenizer with the training data as the words from the short story
text = ("It's the last he painted, you know," 
        Mrs. Gisburn said with pardonable pride.) #Sample text for the tokenizer
ids = tokenizer.encode(text) #Tokenizes
print(ids)
print(tokenizer.decode(ids)) #Converts back 
"""
text = "Hello, do you like tea?"
#print(tokenizer.encode(text)) Returns an error because Hello was not in the short story, which means that the tokenizer wont work because its unseen data

all_tokens = sorted(list(set(preprocessed_data)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"]) #added two new tokens to the previous list of tokens
vocab = {token:integer for integer, token in enumerate(all_tokens)}
print(len(vocab.items()))

for i, item in enumerate(list(vocab.items())[-5:]): #Checking to make sure that the new tokens were properly added
    print(item)
""" Simple Tokenizer V2
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}
    
    def encode(self, text):
        preprocessed_data = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed_data = [item.strip() for item in preprocessed_data if item.strip()]
        preprocessed_data = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed_data]
        ids = [self.str_to_int[s] for s in preprocessed_data]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
    
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
#print(text)
tokenizerV2 = SimpleTokenizerV2(vocab)
print(tokenizerV2.encode(text))
print(tokenizerV2.decode(tokenizerV2.encode(text)))
"""

#Tokenization using the Tiktoken public tokenizer - Uses Byte Pair Encoding
tiktoken_tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    " of someunknownPlace." 
)
integers = tiktoken_tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

strings = tiktoken_tokenizer.decode(integers)
print(strings)


#Shows how a simple dataloader words for the training of an LLM, example of a sliding window.
enc_text = tiktoken_tokenizer.encode(raw_text)
print(len(enc_text))

enc_sample = enc_text[50:]
context_size = 4 #Determines how many tokens are included in the input
x = enc_sample[:context_size] #Input tokens
y = enc_sample[1:context_size + 1] #target tokens
print(f"x: {x}")
print(f"y:      {y}")

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tiktoken_tokenizer.decode(context), "---->", tiktoken_tokenizer.decode([desired]))


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt) #Tokenizes the entire text
        
        #Uses a Sliding Window approach to chunk the text into segments of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length] 
            target_chunk = token_ids[i + 1: i + max_length + 1] 
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
        
    def __len__(self): #Returns the totel number of rows in the dataset
        return len(self.input_ids)
    def __getitem__(self, idx): #Returns a single row of the dataset
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length=max_length, stride=stride)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader
"""
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("Targets:\n", targets)

data_iter = iter(dataloader)
first_batch = next(data_iter)
second_batch = next(data_iter)
print(first_batch)
print(second_batch)
"""

#Example for embedding tokens
"""
input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 6
output_dim = 3
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)
print(embedding_layer(input_ids))
"""

vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length=4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape) #For this specific example of batch_size 8 and max_len of 4, the result is [8,4]: 
                                         #eight text samples, 4 tokens each
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)