# 09-2 | 순환 신경망으로 IMDB 리뷰 분류하기

```python
from tensorflow.keras.datasets import imdb
(train_input, train_target), (test_input, test_target) = imdb.load_data(
    num_words=500)

from sklearn.model_selection import train_test_split
train_input, val_input, train_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)

import numpy as np
lengths = np.array([len(x) for x in train_input])

import matplotlib.pyplot as plt
plt.hist(lengths)
plt.xlabel('length')
plt.ylabel('frequency')
plt.show()
```

![Untitled](https://user-images.githubusercontent.com/87055471/129478805-1e42daa1-7984-4c54-bea1-eb9976b4c6d9.png)


```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
train_seq = pad_sequences(train_input, maxlen=100)

val_seq = pad_sequences(val_input, maxlen=100)

from tensorflow import keras
model = keras.Sequential()
model.add(keras.layers.SimpleRNN(8, input_shape=(100, 500)))
model.add(keras.layers.Dense(1, activation='sigmoid'))

train_oh = keras.utils.to_categorical(train_seq)

val_oh = keras.utils.to_categorical(val_seq)

model.summary()

rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model.compile(optimizer=rmsprop, loss='binary_crossentropy',
              metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-simplernn-model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)
history = model.fit(train_oh, train_target, epochs=100, batch_size=64,
                    validation_data=(val_oh, val_target),
                    callbacks=[checkpoint_cb, early_stopping_cb])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
```
![Untitled 1](https://user-images.githubusercontent.com/87055471/129478809-6b0108dc-81c2-4cfa-92f6-9aefb766cdf2.png)


```python
model2 = keras.Sequential()
model2.add(keras.layers.Embedding(500, 16, input_length=100))
model2.add(keras.layers.SimpleRNN(8))
model2.add(keras.layers.Dense(1, activation='sigmoid'))

model2.summary()

rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model2.compile(optimizer=rmsprop, loss='binary_crossentropy',
               metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-embedding-model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)
history = model2.fit(train_seq, train_target, epochs=100, batch_size=64,
                     validation_data=(val_seq, val_target),
                     callbacks=[checkpoint_cb, early_stopping_cb])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
```
![Untitled 2](https://user-images.githubusercontent.com/87055471/129478817-187a2184-0a40-4d97-9534-2e4b628011f1.png)


순환 신경망의 개념을 실제 모델을 만들어 보면서 구체화해 보았다. 텐서플로와 케라스는 완전 연결 신경망, 합성곱 신경망뿐만 아니라 다양한 순환층 클래스를 제공하기 때문에 손쉽게 순환 신경망을 만들 수 있다.

순환 신경망의 MNIST 데이터셋으로 생각할 수 있는 유명한 IMDB 리뷰 데이터셋을 사용했다. 이 작업은 리뷰의 감상평을 긍정과 부정으로 분류하는 이진 분류 작업이다.

두 가지 모델을 훈련해 보았다. 먼저 입력 데이터를 원-핫 인코딩으로 변환하여 순환층에 직접 주입하는 방법을 사용했다. 두 번째는 정수 시퀀스를 그대로 사용하기 위해 모델 처음에 Embedding 층을 추가했다. 단어 임베딩은 단어마다 실수로 이루어진 밀집 벡터를 학습하기 때문에 단어를 풍부하게 표현할 수 있다.

- **말뭉치**는 자연어 처리에서 사용하는 텍스트 데이터의 모음, 즉 훈련 데이터셋을 일컫는다.
- **토큰**은 텍스트에서 공백으로 구분되는 문자열을 말한다. 종종 소문자로 변환하고 구둣점은 삭제한다.
- **원-핫 인코딩**은 어떤 클래스에 해당하는 원소만 1이고 나머지는 모두 0인 벡터이다. 정수로 변환된 토큰을 원-핫 인코딩으로 변환하면 어휘 사전 크기의 벡터가 만들어진다.
- **단어 임베딩**은 정수로 변환된 토큰을 비교적 작은 크기의 실수 밀집 벡터로 변환한다. 이런 밀집 벡터는 단어 사이의 관계를 표현할 수 있기 때문에 자연어 처리에서 좋은 성능을 발휘한다.
