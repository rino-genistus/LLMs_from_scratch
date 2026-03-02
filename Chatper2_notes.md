**Chapter 2: Working with Text Data**
- Embedding: Converting data into a vector format
- For the training data for the LLM, we are using "The Verdict" a short story by Edith Wharton.
    - Goal is to tokenize this entire text file into individual words and special characters and then turn them into embeddings for LLM training
- First, we are simply looking at sample texts and figuring out how to properly split the text to include what we want
- After loading the short story in, we split all the text by characters and spaces
- Now that the tokens are created, we need to create token IDs. For that, something simple could work like assinging each of them numbers
- For encoding that would work. For example, in the sentence, "Hello World. This is Rino.", "Hello" would be 0, "World" would be 1 and so on. We store this in a dictionary.
- For decoding, we create an inverse of the dictionary and this will allow us to decode the token IDs and form a proper sentence with that.
- The way this works is it basically fetches the word from the dictionary that we have and join them into a proper sentence. 

- New Tokenizer:
    - In the old tokenizer, the tokenizer was unable to use unknown words that it wasn't already trained on. For example, the word Hello was not present in the short story. So when trying to tokenize the sentence with the unknown word in that, it would throw an error because the tokenizer was unable to find the word, integer pair in the dictionary. So in the new tokenizer, so help with this, we added two new tokens to identify the end of the text and for unknown words. 