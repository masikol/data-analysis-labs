from keras.callbacks import ModelCheckpoint, TensorBoard
import pandas as pd
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
import re
import preprocessor as p

twitter_api_key = '9bjRdeeugv2FkMTPQmcDRZAdv'
twitter_api_secret = 'z0iaQR7xoejmGi8ElvA65C0hs6M9IhMgQCWSvy5sRjx3iYp02E'
twitter_bearer_token = f'AAAAAAAAAAAAAAAAAAAAAKi4WAEAAAAAQG65Ns9pXb%2F2SuQXCMEjE5dgGck%3D37luuxlRj1KLBZWBaPQlZ8BX1fauKDluH57vbJDIdahDFF7rte'
access_token = '927121523739103232-j1LdP7Jj9CsWvK84byv4TLLStgLMj8p'
access_token_secret = 'oTYCNp4KYB5yvKYgAkQPiX6uJ3NeWh6dRZ2D0UsY84l2A'
musk_id = 44196397
spears_id = 16409683

REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\|)|(\()|(\))|(\[)|(\])|(\%)|(\$)|(\>)|(\<)|(\{)|(\})")
REPLACE_WITH_SPACE = re.compile("(<br\s/><br\s/?)|(-)|(/)|(:).")

def clean_tweets(df):
  """
  Функция для очистки текстов твитов от спец. символов, ссылок, ников других пользоваталей, смайликов.
  """

  tempArr = []
  for line in df:
    tmpL = p.clean(line)
    tmpL = REPLACE_NO_SPACE.sub("", tmpL.lower())
    tmpL = REPLACE_WITH_SPACE.sub(" ", tmpL)
    tempArr.append(tmpL)
  return tempArr

def main():
    raw_data = pd.read_csv('/home/akhlebko/Workspace/sentiment-python/files/data/training_data.csv', header=None)
    raw_val_data = pd.read_csv('/home/akhlebko/Workspace/sentiment-python/files/data/test_data.csv', header=None)

    num_of_words = 400000 # количество слов в тексте было получено опытным путем
    tokenizer = Tokenizer(num_words=num_of_words, oov_token='<OOV>')
    num_of_texts = 400000 # количество твиттов, используемых для обучения метода
    feature_values = [str(x).strip() for x in raw_data[6].values[0:num_of_texts]]
    validation_features = [str(x).strip() for x in raw_val_data[6].values]

    # препроцессинг текстов и показателя сентиментальности
    tokenizer.fit_on_texts(feature_values)
    num_embedd = len(tokenizer.word_index) + 1
    val_sequences = pad_sequences(tokenizer.texts_to_sequences(validation_features), maxlen=140, truncating='post')
    sequences = tokenizer.texts_to_sequences(feature_values)
    train_X = pad_sequences(sequences, maxlen=140, truncating='post')
    Y_data = []
    for y in raw_data[0].values[:num_of_texts]:
        if (y == 0):
            Y_data.append([1, 0])
        if (y == 4):
            Y_data.append([0, 1])

    val_Y_data = []
    for y in raw_val_data[0].values:
        if (y == 0):
            val_Y_data.append([1, 0])
        if (y == 4):
            val_Y_data.append([0, 1])

    # создание модели
    embed_dim = 128
    lstm_out = 196
    model = Sequential()
    model.add(Embedding(num_embedd, embed_dim,input_length = train_X.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    print(model.summary())

    # коллбэки для трекинга обучения и сохранения чекпоинтов    
    checkpoint = ModelCheckpoint("best_model.hdf5", monitor='val_accuracy', verbose=1, mode='auto', period=1,save_weights_only=False)
    tensorboard_callback = TensorBoard(log_dir="./logs", write_steps_per_second=True)

    # обучение модели
    batch_size = 32
    model.fit(
        np.array(feature_values), np.array(Y_data), 
        validation_data=(np.array(validation_features, val_Y_data)),
        validation_batch_size=1,
        epochs = 7,
        batch_size=batch_size,
        verbose = 1,
        callbacks=[checkpoint, tensorboard_callback]
    )

    # евалуация модели
    score = model.evaluate(np.array(val_sequences), np.array(val_Y_data), verbose=1, return_dict=True)
    print(score)
    del model

if __name__ == "__main__":
    main()