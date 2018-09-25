import pandas as pd
from keras.preprocessing import text

df = pd.read_csv('train_pos.tab',sep='\t')
df.loc[df['emotion']=='6','emotion']='neutral'
df['label'] = 0
df.loc[df['emotion']=='positive','label'] = 1

df = df.sample(1000)

tdf = pd.read_csv('dev_pos.tab',sep='\t')
tdf.loc[tdf['emotion']=='6','emotion']='neutral'
tdf['label'] = 0
tdf.loc[tdf['emotion']=='positive','label'] = 1

tdf = tdf.sample(100)

t = text.Tokenizer()
t.fit_on_texts(df['utterance'])
x_train = t.texts_to_matrix(df['utterance'],mode='count')
y_train = df['label']
x_test = t.texts_to_matrix(tdf['utterance'],mode='count')
y_test = tdf['label']

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

model = Sequential()
model.add(Embedding(x_train.shape[0], output_dim=64))
model.add(LSTM(32))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=2)
score = model.evaluate(x_test, y_test, batch_size=16)

