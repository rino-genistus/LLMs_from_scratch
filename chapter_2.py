import re

with open("/Users/rino/LLMs_From_Scratch/LLMs_from_scratch/the-verdict.txt", "r", encoding="utf-8") as f: #Imports the text file into the python file
    raw_text = f.read()
print("Total Number of characters: ", len(raw_text)) #reads the total number of characters in the text
print(raw_text[:99]) # Prints the first hundred characters of the text

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
text = """"It's the last he painted, you know," 
        Mrs. Gisburn said with pardonable pride.""" #Sample text for the tokenizer
ids = tokenizer.encode(text) #Tokenizes
print(ids)
print(tokenizer.decode(ids)) #Converts back

text = "Hello, do you like tea?"
#print(tokenizer.encode(text)) Returns an error because Hello was not in the short story, which means that the tokenizer wont work because its unseen data

all_tokens = sorted(list(set(preprocessed_data)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"]) #added two new tokens to the previous list of tokens
vocab = {token:integer for token, integer in enumerate(all_tokens)}
print(len(vocab.items()))

for i, item in enumerate(list(vocab.items())[-5:]): #Checking to make sure that the new tokens were properly added
    print(item)

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
