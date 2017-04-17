import sys, os
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Lambda, TimeDistributed, Dense, BatchNormalization, Merge, Concatenate
from keras.layers.merge import concatenate
from keras import backend as K
from sklearn.model_selection import train_test_split

from dec_att import build_model

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
nb_epoch=10

df=pd.read_csv(os.path.join(data_dir, 'train.csv'))
df['question1']=df['question1'].apply(str)
df['question2']=df['question2'].apply(str)

questions=list(df['question1'])+list(df['question2'])
tokenizer=Tokenizer(max_nb_words)
tokenizer.fit_on_texts(questions)
q1_seqs=tokenizer.texts_to_sequences(list(df['question1']))
q2_seqs=tokenizer.texts_to_sequences(list(df['question2']))
word_index=tokenizer.word_index

nb_words=min(max_nb_words, len(word_index))
embedding_matrix=pickle.load(open('glovemat840B.300d.pickle', 'rb'))
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

q1_data=pad_sequences(q1_seqs, maxlen=max_seq_length)
q2_data=pad_sequences(q2_seqs, maxlen=max_seq_length)
labels=np.asarray(df['is_duplicate'], dtype=int)
print('Q1 shape: ', q1_data.shape)
print('Q2 shape: ', q2_data.shape)
print('Labels shape: ', labels.shape)

settings={'lr':0.001, 'dropout':0.2, 'gru_encode':False}
model=build_model(embedding_matrix, (max_seq_length, embedding_dim, 1), settings)

for i in range(nb_epoch):
	hist=model.fit([q1_data, q2_data], labels, batch_size=64, epochs=1, validation_split=val_split)
	model.save_weights("W{}-{}.h5".format(i, hist.history))
