#各種モジュール(Keras/NumPy/Matplotlib)のインポート
from __future__ import print_function

from tensorflow  import keras
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop

import numpy as np
import matplotlib.pyplot as plt


#データセットの取得(訓練データとテストデータの分割)
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


#データの内容確認
print("\n")
print("--データの内容確認--")
print(x_train.shape) #(60000, 28, 28)
print(x_test.shape) #(10000, 28, 28)
print(y_train.shape) #(60000,)
print(y_test.shape) #(10000,)
print("\n")

#データの整形
#Numpyのreshameメゾットを利用
x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


#データの正規化
x_train /= 255
x_test /= 255


#テストデータの内容確認
print("\n")
print("--x_train[0]--")
print("\n")

for a in range(28):
    w = ""
    for b in range(28):
        w_next = "{0:.0f}".format(x_train[0][28 * a + b]) + " " #format関数 "{0:指定したい書式の型}".format(変数)
        w += w_next
    print(w)

print("\n")


#訓練データの整形
y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)


#ニューラルネットワークの実装①
#今回のニューラルネットワーク：入力層 → 隠れ層① → 隠れ層② → 出力層
#relu:Relu関数/softmax:softmax関数
#Dropout:ドロップアウト（過学習を防ぐために設定 → 推論の精度が向上する）

##入力層
model = Sequential()

##隠れ層①
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))

##隠れ層②
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

##出力層
model.add(Dense(10, activation='softmax'))

##要約の出力
model.summary()
print("\n")


#ニューラルネットワークの実装②
#loss:損失関数の設定（categorical_crossentropy:クロスエントロピー）
#optimizer:最適化アルゴリズムの設定(RMSprop:リカレントニューラルネットワークに対して設定するのが良いらしい)
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])


#ニューラルネットワークの学習
#epochs:ニューラルネットワークに何回学習させるか
history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=20,
                    verbose=1,
                    validation_data=(x_test, y_test))


#ニューラルネットワークの推論
score = model.evaluate(x_test,y_test,verbose=0)
print("\n")
print("Test loss:",score[0])
print("Test accuracy:",score[1])
print("\n")


#ニューラルネットワークの推論があっているか確認
##推論した結果(数字)とファッションアイテム名は、下記のitemsのように設定されている
def fashion_classify(predict):
    items = {0:"Tシャツ/トップス",1:"ズボン",2:"プルオーバー",3:"ドレス",4:"コート",5:"サンダル",6:"シャツ",7:"スニーカー",8:"バック",9:"アンクルブーツ"}

    return items[int(predict)]

##結果表示
print("--x_test[0]の推論結果--")
predict_prob=model.predict(x_test[0].reshape(1,-1),batch_size=1,verbose=0)
predict_classes=np.argmax(predict_prob,axis=1)
predict = predict_classes
print(fashion_classify(predict))
print("----")
print("\n")


#画像を表示
img = x_test[0].reshape(28,28)
plt.imshow(img)
plt.show()