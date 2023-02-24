import spacy
nlp = spacy.load('en_core_web_md')

#Closest similarity between cat/monkey as animals. Monkey/banana had a higher similarity because it recognised monkeys eat bananas. 
print("=====Similarity between three words======")
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")
print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

print("=====Similarity of 3 new words======")
word1 = nlp("granite")
word2 = nlp("mountain")
word3 = nlp("coal")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word2))
#very interesting that it recognises a closer relationship between mountain and granite (because thats where you find granite).
#but mountain and coal does not trigger the same similarity. 

#Recognised the similarities of each word in a list with each other. 
tokens = nlp("cat apple monkey banana")

print("=======Similarity of each word in a list with each other======")
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))


print("======Simiarlity between sentences=====")

sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go",
"Hello, there is my car",
"I\'ve lost my car in my car",
"I\'d like my boat back",
"I will name my dog Diana"]
model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)


#Differences when changing language module in example file:
#With the simpler language model there are no word vectors downloaded. This limits the similarity link that the program can make between words. 
#For example, this means 'recipe similarity' has a drop of almost 0.2 when comparing recipe vs complaint. 

# This was the userwarning prompted: "The model you're using has no word vectors loaded, so the result of the Doc.similarity method will be based on the tagger, 
# parser and NER, which may not give useful similarity judgements. 
#This may happen if you're using one of the small models, e.g. `en_core_web_sm`, 
# which don't ship with word vectors and only use context-sensitive tensors. 
# You can always add your own word vectors, or use one of the larger models instead if available.""