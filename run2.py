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
nb_epoch=4
nb_start=20

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

Q1=Sequential()
Q1.add(Embedding(nb_words+1, 
                 embedding_dim, 
                 weights=[embedding_matrix], 
                 input_length=max_seq_length, 
                 trainable=True))
Q1.add(TimeDistributed(Dense(embedding_dim, activation='relu')))
Q1.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(embedding_dim, )))

Q2=Sequential()
Q2.add(Embedding(nb_words+1, 
                 embedding_dim, 
                 weights=[embedding_matrix], 
                 input_length=max_seq_length, 
                 trainable=True))
Q2.add(TimeDistributed(Dense(embedding_dim, activation='relu')))
Q2.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(embedding_dim, )))

model=Sequential()
model.add(Merge([Q1, Q2], mode='concat'))
model.add(BatchNormalization())
model.add(Dense(200,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(200,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(200,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1,activation='sigmoid'))

loss='binary_crossentropy'
optimizer='adam'
metrics=['accuracy']
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
model.load_weights("W20-{'val_loss': [0.36860401125187048], 'val_acc': [0.82876153257505347], 'loss': [0.33473091030628271], 'acc': [0.8474362462596926]}.h5")
for i in range(nb_start, nb_start+nb_epoch):
	hist=model.fit([q1_data, q2_data], labels, batch_size=64, epochs=1, validation_split=val_split)
	model.save_weights("W{}-{}.h5".format(i, hist.history))
