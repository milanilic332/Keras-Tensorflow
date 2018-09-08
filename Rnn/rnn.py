import numpy as np
import os
from copy import copy
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.optimizers import Adam

# Used characters in c files
charset = set()

# Number of files used as input (Limited RAM)
max_files = 200

# Make charset
for i, file in enumerate(os.listdir('data/c_files')):
    if i == max_files:
        break
    with open(os.path.join('data/c_files', file), 'r', encoding='UTF-8') as f:
        text = f.read().lower()
        for t in text: charset.add(t)
        i += 1

charset = list(charset)

# Go from int to char and vice versa
char_to_int = dict((c, i) for i, c in enumerate(charset))
int_to_char = dict((i, c) for i, c in enumerate(charset))

# Length of input layer
seq_len = 80

# Making inputs and labels
sen = []
next = []
for i, file in enumerate(os.listdir('data/c_files')):
    if i == max_files:
        break
    with open(os.path.join('data/c_files', file), 'r', encoding='UTF-8') as f:
        text = f.read().lower()
        for i in range(len(text) - seq_len):
            sen.append([char_to_int[c] for c in text[i:i + seq_len]])
            next.append(char_to_int[text[i + seq_len]])
    i += 1

n_patterns = len(sen)

print('Length of charset: %d' % len(charset))
print("Number of patterns: %d" % n_patterns)
print(charset)

# Preparing the data
X = np.reshape(sen, (n_patterns, seq_len, 1))
X = X/float(len(charset))
Y = np_utils.to_categorical(next)

# Model architecture
model = Sequential()

model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(512))
model.add(Dropout(0.2))

model.add(Dense(Y.shape[1], activation='softmax'))

optimizer = Adam(0.001, decay=1e-8)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Writing the 1000 character long prediction after each epoch
def on_epoch_end(epoch, logs):
    pattern = copy(sen[0])
    with open('data/result-after-' + str(epoch) + '-epoch80.txt', 'w+') as res:
        for i in pattern:
            res.write(int_to_char[i])
        for i in range(1000):
            x = np.reshape(pattern, (1, len(pattern), 1))
            x = x/float(len(charset))

            prediction = model.predict(x, verbose=0)
            index = np.argmax(prediction)
            result = int_to_char[index]

            res.write(result)

            pattern.append(index)
            pattern = pattern[1:len(pattern)]

    print("Done.")

# Saving model after each epoch
filepath = "models/weight-{epoch:02d}-{loss:.4f}80.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

callbacks_list = [checkpoint, print_callback]

model.fit(X, Y, epochs=50, batch_size=320, callbacks=callbacks_list)
