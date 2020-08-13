import numpy as np
from tensorflow import keras
#from keras.layers.core import Dense, Activation, Dropout
#from keras.layers import LSTM
#from keras.models import Sequential, load_model
#from keras.callbacks import Callback
#import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import pandas as pd
import os
#import keras.callbacks
import matplotlib.pyplot as plt

class LSTM_createModel:
    def __init__(self,usernum=181):
        self.usernum = usernum
        # 设定为自增长
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.session = tf.Session(config=self.config)
        keras.backend.set_session(self.session)


    def create_dataset(self,data, n_predictions, n_next):
        '''
        对数据进行处理
        '''
        dim = data.shape[1]
        train_X, train_Y = [], []
        for i in range(data.shape[0] - n_predictions - n_next - 1):
            a = data[i:(i + n_predictions), :]
            train_X.append(a)
            tempb = data[(i + n_predictions):(i + n_predictions + n_next), :]
            b = []
            for j in range(len(tempb)):
                for k in range(dim):
                    b.append(tempb[j, k])
            train_Y.append(b)
        train_X = np.array(train_X, dtype='float64')
        train_Y = np.array(train_Y, dtype='float64')

        test_X, test_Y = [], []
        i = data.shape[0] - n_predictions - n_next - 1
        a = data[i:(i + n_predictions), :]
        test_X.append(a)
        tempb = data[(i + n_predictions):(i + n_predictions + n_next), :]
        b = []
        for j in range(len(tempb)):
            for k in range(dim):
                b.append(tempb[j, k])
        test_Y.append(b)
        test_X = np.array(test_X, dtype='float64')
        test_Y = np.array(test_Y, dtype='float64')

        return train_X, train_Y, test_X, test_Y


    def NormalizeMult(self,data, set_range):
        '''
        返回归一化后的数据和最大最小值
        '''
        normalize = np.arange(2 * data.shape[1], dtype='float64')
        normalize = normalize.reshape(data.shape[1], 2)

        for i in range(0, data.shape[1]):
            if set_range == True:
                list = data[:, i]
                listlow, listhigh = np.percentile(list, [0, 100])
            else:
                if i == 0:
                    listlow = -90
                    listhigh = 90
                else:
                    listlow = -180
                    listhigh = 180

            normalize[i, 0] = listlow
            normalize[i, 1] = listhigh

            delta = listhigh - listlow
            if delta != 0:
                for j in range(0, data.shape[0]):
                    data[j, i] = (data[j, i] - listlow) / delta

        return data, normalize


    def trainModel(self,train_X, train_Y):
        '''
        trainX，trainY: 训练LSTM模型所需要的数据
        '''
        model = keras.models.Sequential()
        model.add(keras.layers.LSTM(
            120,
            input_shape=(train_X.shape[1], train_X.shape[2]),
            return_sequences=True))
        model.add(keras.layers.Dropout(0.3))

        model.add(keras.layers.LSTM(
            120,
            return_sequences=False))
        model.add(keras.layers.Dropout(0.3))

        model.add(keras.layers.Dense(
            train_Y.shape[1]))
        model.add(keras.layers.Activation("relu"))

        model.compile(loss='mse', optimizer='adam', metrics=['acc'])
        model.fit(train_X, train_Y, epochs=100, batch_size=64, verbose=1)
        model.summary()

        return model

    def train_and_save(self,userid,train_num=6,pred_num=1,draw_pic=True):
        per_num = pred_num
        # set_range = False
        set_range = True

        # 读入时间序列的文件数据
        data = np.empty((0, 2))
        #print(data.shape)
        filenum = 0
        maxfilenum = 10  #控制读取文件的数量，进而控制样本数
        for i, j, filenamelst in os.walk('./Geolife Trajectories 1.3/Data/' + userid + '/Trajectory/'):
            for filename in filenamelst:
                filenum += 1
                tmpdata = pd.read_csv('./Geolife Trajectories 1.3/Data/' + userid + '/Trajectory/' + str(filename),
                                      sep=',',
                                      header=6).iloc[:,
                          0:2].values
                data = np.concatenate((data, tmpdata), axis=0)
                if filenum == maxfilenum:
                    break
        print("样本数：{0}，维度：{1}".format(data.shape[0], data.shape[1]))
        print(data)
        if draw_pic:
            # 画样本数据库
            plt.scatter(data[:, 1], data[:, 0], c='b', marker='o', label='traj_A')
            plt.legend(loc='upper left')
            plt.grid()
            plt.show()

        # 归一化
        data, normalize = self.NormalizeMult(data, set_range)
        print(normalize)

        # 生成训练数据
        train_X, train_Y, test_X, test_Y = self.create_dataset(data, train_num, per_num)
        print("x\n", train_X.shape)
        print("y\n", train_Y.shape)

        # 训练模型
        model = self.trainModel(train_X, train_Y)
        loss, acc = model.evaluate(train_X, train_Y, verbose=2)
        print('Loss : {}, Accuracy: {}'.format(loss, acc * 100))

        # 保存模型
        if not os.path.exists('npy'):
            os.mkdir(os.getcwd() + '\\npy')
        np.save("./npy/"+userid+"_traj_model_trueNorm.npy", normalize)
        model.save("./model/"+userid+"_traj_model.h5")

    def train_every_user(self):
        for i in range(0,10):
            userid = str(i).zfill(3)
            self.train_and_save(userid=userid,train_num=6,pred_num=1,draw_pic=False)

if __name__ == "__main__":
    newmodel = LSTM_createModel(181)
    newmodel.train_every_user()
    
