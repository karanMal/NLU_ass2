from keras.models import load_model
from pickle import load
from pickle import dump
from keras.preprocessing.text import Tokenizer
import numpy as np
import warnings
warnings.filterwarnings('ignore')



# load the model
model = load_model('model.h5')

# load the tokenizer
tokenizer = load(open('token.pkl', 'rb'))

#  function to sample from multinomial distribution
def sample_multi(preds):
    preds = np.asarray(preds).astype('float64');preds=np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)




# function to generate random sentence
def generate_seq(model, tokenizer,seed_text, n_words):
    result=[]
    text=seed_text
    for i in range(n_words):
      
      sequence=tokenizer.texts_to_sequences([text])[0]
      yhat=model.predict(sequence,verbose=0)
      word_index=sample_multi(yhat[-1,0,:])
      output=''
      for word, index in tokenizer.word_index.items():
            if index==word_index:
                output=word
                break
      result.append(output);#print(output)
      text.append(output);#print(text)
	  
	  
    return result
      

initial_words=['my','his','they','her','the','we','i']
temp =np.random.choice(len(initial_words),1)
input_seq=initial_words[temp[0]].split()
# generate new sentence
generated=generate_seq(model, tokenizer, input_seq,14)
generated.insert(0,input_seq[0])
print(' '.join(generated))
