import nltk
from nltk.corpus import  gutenberg
from collections import Counter
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from numpy import array
import numpy as np
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Activation

def model_building(embedding,lstm1,lstm2,dense1,vocab_len,seq_length):

    model=Sequential()
    model.add(Embedding(vocab_len,embedding))  # input dimension, output dimension,input sequence length
    model.add(LSTM(lstm1,return_sequences=True))
    #model.add(LSTM(lstm2,return_sequences=True))
    model.add(Dense(dense1,activation='relu'))
    model.add(Dense(vocab_len,activation='softmax'))
    
    print(model.summary())
    return model



  
# preparing data
total_sent1=list(gutenberg.sents('austen-emma.txt'))
total_sent2=list(gutenberg.sents('austen-persuasion.txt'))
total_sent3=list(gutenberg.sents('bryant-stories.txt'))
total_sent4=list(gutenberg.sents('whitman-leaves.txt'))
total_sent5=list(gutenberg.sents('milton-paradise.txt'))
total_sent6=list(gutenberg.sents('carroll-alice.txt'))


train_index1=int(0.8*len(total_sent1))
train_index2=int(0.8*len(total_sent2))
train_index3=int(0.8*len(total_sent3))
train_index4=int(0.8*len(total_sent4))
train_index5=int(0.8*len(total_sent5))
train_index6=int(0.8*len(total_sent6))


# 80:20 split
train= total_sent1[0: train_index1]+total_sent2[0: train_index2]+total_sent3[0: train_index3]+total_sent4[0: train_index4]+total_sent5[0: train_index5]+total_sent6[0: train_index6]
test= total_sent1[train_index1:]+total_sent2[train_index2:]+total_sent3[train_index3:]+total_sent4[train_index4:]+total_sent5[train_index5:]+total_sent6[train_index6:]


# In[45]:



#converting words into sequences 
train_words=[]
for i in range(len(train)):
    for word in train[i]:
        train_words.append(word.lower())


# cleaning of words
train_words=[words for words in train_words if words.isalpha()]
#print(len(train_words))

vocab=Counter(train_words)
#print(len(vocab))


#joining of 51 words as a sequence.

sequence=[]
k=0
window=51
stride=1
#applying windowing for sequence genration

for i in range(0,len(train_words)-window,stride):
    line=train_words[i:i+window]
    sequence.append(' '.join(line))
    sequence[k]=sequence[k].split()
    k+=1


#labeling the words in sequences.....................
tokenizer=Tokenizer()
tokenizer.fit_on_texts(sequence)
seq=tokenizer.texts_to_sequences(sequence)
vocab_len=len(tokenizer.word_index.items())+1
#print(vocab_len)




    # one hot encoding of y
seq=array(seq)
x_train=seq[:,:-1]
y_train=np.zeros((x_train.shape[0],x_train.shape[1],1))
for i in range(x_train.shape[0]):
  for j in range(x_train.shape[1]):
    y_train[i,j,0]=seq[i,j+1]

    
#print(x_train.shape)
#print(y_train.shape)

# In[46]:



seq_length= x_train.shape[1]

LM=model_building(30,50,50,50,vocab_len,seq_length)
LM.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
LM.fit(x_train,y_train,batch_size=256,epochs=50)



# saving model
LM.save('model.h5')
dump(tokenizer, open('token.pkl', 'wb'))

#Evaluation of model starts

# Function for Calculating Perplexity
def perplexity(x_test,y_test):
	N=x_test.shape[0]*x_test.shape[1]
	perp=1
	for i in range(x_test.shape[0]):
		for j in range(x_test.shape[1]):
			yhat=model.predict(x_test[i].reshape(1,-1),verbose=0)
			pred=yhat[-1,0,:]
			if pred[int(y_test[i,j,0])]>0.00001:
				prob= 1/pred[int(y_test[i,j,0])]
				prob = (prob)**(1/N)
				perp*=prob
		


	# In[49]:

# Building of Test data sequences
#converting words into sequences 
test_words=[]
for i in range(len(test)):
    for word in test[i]:
        test_words.append(word.lower())


# cleaning of words
test_words=[words for words in test_words if words.isalpha()]
#print(len(test_words))

#joining of 51 words as a sequence.

sequence=[]
k=0
window=51
stride=1

#applying windowing for sequence genration

for i in range(0,len(test_words)-window,stride):
    line=test_words[i:i+window]
    sequence.append(' '.join(line))
    sequence[k]=sequence[k].split()
    k+=1


#labeling the words in sequences...................
seq=tokenizer.texts_to_sequences(sequence)


seq=array(seq)
x_test=seq[:,:-1]
y_test=np.zeros((x_test.shape[0],x_test.shape[1],1))
for i in range(x_test.shape[0]):
  for j in range(x_test.shape[1]):
    y_test[i,j,0]=seq[i,j+1]

p=perplexity(x_test,y_test)
print(p)





