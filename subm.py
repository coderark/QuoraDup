import sys, os
import pandas as pd
import numpy as np
import spacy
import pickle
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Lambda, TimeDistributed, Dense, BatchNormalization, Merge, Concatenate
from keras.layers.merge import concatenate
from keras import backend as K
from sklearn.model_selection import train_test_split

from models import model1, model2

cur_dir=os.getcwd()
data_dir=os.path.join(cur_dir, 'data')
save_dir=os.path.join(data_dir, 'save')
model_dir=os.path.join(os.path.dirname(cur_dir), 'models')
glove_file=os.path.join(model_dir, 'glove/glove.6B.300d.txt')
if not os.path.exists(save_dir): os.mkdir(save_dir)
    
max_nb_words=200000
max_seq_length=25
embedding_dim=300
val_split=0.1
test_split=0.1

df=pd.read_csv(os.path.join(data_dir, 'train.csv'))
dft=pd.read_csv(os.path.join(data_dir, 'test.csv'))
df['question1']=df['question1'].apply(str)
df['question2']=df['question2'].apply(str)
dft['question1']=dft['question1'].apply(str)
dft['question2']=dft['question2'].apply(str)
test_id=np.array(dft['test_id'])

questions=list(df['question1'])+list(df['question2'])
tokenizer=Tokenizer(max_nb_words)
tokenizer.fit_on_texts(questions)
q1_seqs=tokenizer.texts_to_sequences(list(df['question1']))
q2_seqs=tokenizer.texts_to_sequences(list(df['question2']))
tq1_seqs=tokenizer.texts_to_sequences(list(dft['question1']))
tq2_seqs=tokenizer.texts_to_sequences(list(dft['question2']))
word_index=tokenizer.word_index

nb_words=min(max_nb_words, len(word_index))
embedding_matrix=pickle.load(open('glovemat840B.300d.pickle', 'rb'))
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

q1_data=pad_sequences(q1_seqs, maxlen=max_seq_length)
q2_data=pad_sequences(q2_seqs, maxlen=max_seq_length)
tq1_data=pad_sequences(tq1_seqs, maxlen=max_seq_length)
tq2_data=pad_sequences(tq2_seqs, maxlen=max_seq_length)
labels=np.asarray(df['is_duplicate'], dtype=int)
print('Q1 shape: ', q1_data.shape)
print('Q2 shape: ', q2_data.shape)
print('Labels shape: ', labels.shape)

#m1=model1(embedding_matrix, max_seq_length, nb_words)
m2=model2(embedding_matrix, max_seq_length, nb_words)

loss='binary_crossentropy'
optimizer='adam'
metrics=['accuracy']
#m1.compile(loss=loss, optimizer=optimizer, metrics=metrics)
m2.compile(loss=loss, optimizer=optimizer, metrics=metrics)
#m1.load_weights("4-0.373832924005.h5")
m2.load_weights("W8-{'val_loss': [0.38170239355064683], 'val_acc': [0.8256449578405709], 'loss': [0.30116606968005971], 'acc': [0.86464061825938154]}.h5")
preds=m2.predict([tq1_data, tq2_data], verbose=True)
preds+=m2.predict([tq2_data, tq1_data], verbose=True)
preds/=2


#dfp=pd.read_csv("s5.csv")
#pred1=df["is_duplicate"]
#preds+=pred1
#preds/=2

out_df = pd.DataFrame({"test_id":test_id, "is_duplicate":preds.ravel()})
out_df.to_csv("s9.csv", index=False)

