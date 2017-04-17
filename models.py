from keras.models import Sequential
from keras.layers import Embedding, Lambda, TimeDistributed, Dense, BatchNormalization, Merge, Concatenate
from keras import backend as K

def model1(embedding_matrix, max_seq_length, nb_words, embedding_dim=300):
	Q1=Sequential()
	Q1.add(Embedding(nb_words+1, 
		         embedding_dim, 
		         weights=[embedding_matrix], 
		         input_length=max_seq_length, 
		         trainable=False))
	Q1.add(TimeDistributed(Dense(embedding_dim, activation='relu')))
	Q1.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(embedding_dim, )))

	Q2=Sequential()
	Q2.add(Embedding(nb_words+1, 
		         embedding_dim, 
		         weights=[embedding_matrix], 
		         input_length=max_seq_length, 
		         trainable=False))
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
	return model

def model2(embedding_matrix, max_seq_length, nb_words, embedding_dim=300):
	Q1=Sequential()
	Q1.add(Embedding(nb_words+1, 
		         embedding_dim, 
		         weights=[embedding_matrix], 
		         input_length=max_seq_length, 
		         trainable=False))
	Q1.add(TimeDistributed(Dense(embedding_dim, activation='relu')))
	Q1.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embedding_dim, )))

	Q2=Sequential()
	Q2.add(Embedding(nb_words+1, 
		         embedding_dim, 
		         weights=[embedding_matrix], 
		         input_length=max_seq_length, 
		         trainable=False))
	Q2.add(TimeDistributed(Dense(embedding_dim, activation='relu')))
	Q2.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embedding_dim, )))

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
	return model
