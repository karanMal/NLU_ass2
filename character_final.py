
# coding: utf-8

# In[1]:


import nltk
from nltk.corpus import gutenberg
from keras.models import Sequential
from keras.layers import Dense, Activation,Embedding
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
from collections import Counter
from keras.preprocessing.text import Tokenizer
from numpy import array
from pickle import dump
from keras.models import load_model
from pickle import load


# In[2]:

#Funcion for model generation
def model_building(embedding,lstm1,lstm2,dense1,vocab_len,seq_length):

    model=Sequential()
    model.add(Embedding(vocab_len,embedding,input_length=seq_length))  # input dimension, output dimension,input sequence length
    model.add(LSTM(lstm1,return_sequences=True))
    model.add(LSTM(lstm2))
    model.add(Dense(dense1,activation='relu'))
    model.add(Dense(vocab_len))
    model.add(Activation('softmax'))
    print(model.summary())
    return model

	

# In[36]:


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




train_chars=[]



# converting into characters

for sentence in train:
    for word in sentence:
        
        for char in word:
            if char!='[' and char!=']':
                
                train_chars.append(char.lower())
        train_chars.append(' ')
    
    
print('Total no of chars %d'%len(train_chars))
    



# In[37]:


#joining of 51 chars as a sequence.

sequence=[]
window=41
stride=7
#applying windowing for sequence genration

for i in range(0,len(train_chars)-window,stride):
    line=train_chars[i:i+window]
    sequence.append(line)

    #labeling the words in sequences.....................
tokenizer=Tokenizer()
tokenizer.fit_on_texts(sequence)
seq=tokenizer.texts_to_sequences(sequence)
vocab_len=len(tokenizer.word_index.items())+1
print(vocab_len)




#
seq=array(seq)
print(seq.shape)
x_train,y_train=seq[:,:-1],seq[:,-1]
#y_train=to_categorical(y_train,num_classes=vocab_len)
y_train=y_train.reshape(-1,1)

print(x_train.shape[0])
seq_length= x_train.shape[1] 


# In[39]:


# model_building

CM=model_building(50,50,50,50,vocab_len,seq_length)
CM.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
CM.fit(x_train,y_train,batch_size=256,epochs=50)



# saving model
CM.save('CM.h5')
dump(tokenizer, open('tokenizer_char.pkl', 'wb'))


# In[57]:
# Function to Calculate Perplexity
def perplexity_cal(x_test,y_test):
    print(x_test[0])
    N=x_test.shape[0]
    perp=1
    for i in range(N):
        pred=CM.predict(x_test[i].reshape(1,-1),verbose=0)[0]
        if pred[y_test[i]]!=0:
            perp *=  1 / ( (pred[y_test[i]]) **(1/N) )




    # In[49]:


    return perp
    #print(gutenberg.fileids())



# load the model
#CM = load_model('CM.h5')
 
# load the tokenizer
#tokenizer = load(open('tokenizer_char.pkl', 'rb'))
vocab_char={}

for char in tokenizer.word_index.items():
    vocab_char[char[0]]=1
test_chars=[]
for sentence in test:
    for word in sentence:
        
        for char in word:
            if char!='[' and char!=']':
                if char in vocab_char:
                    test_chars.append(char.lower())
        test_chars.append(' ')
print('Total no of chars %d'%len(test_chars))


#joining of 51 chars as a sequence.

sequence2=[]
window=41
stride=7
#applying windowing for sequence genration

for i in range(0,len(test_chars)-window,stride):
    line=test_chars[i:i+window]
    sequence2.append(line)
print(len(sequence2[0]))

    #labeling the words in sequences.....................

seq2=tokenizer.texts_to_sequences(sequence2)

vocab_len=len(tokenizer.word_index.items())+1
#print(vocab_len)

#print((seq2[0]))


seq2=array(seq2)


x_test,y_test=seq2[:,:-1],seq2[:,-1]
#y_train=to_categorical(y_train,num_classes=vocab_len)
y_test=y_test.reshape(-1,1)

print(x_test.shape[0])
perp=perplexity_cal(x_test,y_test)
print('Perplexity is coming to be:')
print(perp)







