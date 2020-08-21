import numpy as np
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import os
import matplotlib.pyplot as plt
import copy
from math import sin, asin, cos, radians, fabs, sqrt

class Predict_via_LSTM_model:
    def __init__(self):
        # 设定为自增长
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.session = tf.Session(config=self.config)
        keras.backend.set_session(self.session)

        self.EARTH_RADIUS = 6371  # 地球平均半径，6371km


    def rmse(self,predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())


    def mse(self,predictions, targets):
        return ((predictions - targets) ** 2).mean()


    def reshape_y_hat(self,y_hat, dim):
        re_y = []
        i = 0
        while i < len(y_hat):
            tmp = []
            for j in range(dim):
                tmp.append(y_hat[i + j])
            i = i + dim
            re_y.append(tmp)
        re_y = np.array(re_y, dtype='float64')
        return re_y


    # 多维反归一化
    def FNormalizeMult(self,data, normalize):
        data = np.array(data, dtype='float64')
        # 列
        for i in range(0, data.shape[1]):
            listlow = normalize[i, 0]
            listhigh = normalize[i, 1]
            delta = listhigh - listlow
            #print("listlow, listhigh, delta", listlow, listhigh, delta)
            # 行
            if delta != 0:
                for j in range(0, data.shape[0]):
                    data[j, i] = data[j, i] * delta + listlow

        return data


    # 使用训练数据的归一化
    def NormalizeMultUseData(self,data, normalize):
        for i in range(0, data.shape[1]):

            listlow = normalize[i, 0]
            listhigh = normalize[i, 1]
            delta = listhigh - listlow

            if delta != 0:
                for j in range(0, data.shape[0]):
                    data[j, i] = (data[j, i] - listlow) / delta

        return data

    # 计算两个经纬度之间的直线距离
    def hav(self,theta):
        s = sin(theta / 2)
        return s * s


    def get_distance_hav(self,lat0, lng0, lat1, lng1):
        # "用haversine公式计算球面两点间的距离。"
        # 经纬度转换成弧度
        lat0 = radians(lat0)
        lat1 = radians(lat1)
        lng0 = radians(lng0)
        lng1 = radians(lng1)

        dlng = fabs(lng0 - lng1)
        dlat = fabs(lat0 - lat1)
        h = self.hav(dlat) + cos(lat0) * cos(lat1) * self.hav(dlng)
        distance = 2 * self.EARTH_RADIUS * asin(sqrt(h))
        return distance

    #从文件中读取输入并预测
    def predict_fromFile_and_draw(self,userid, test_num=6, pred_num=1,draw_pic=True):
        per_num = pred_num
        for i,j,filenamelst in os.walk('./Geolife Trajectories 1.3/Data/'+userid+'/Trajectory/'):
            #print(filenamelst[0])
            data_all = pd.read_csv('./Geolife Trajectories 1.3/Data/'+userid+'/Trajectory/'+filenamelst[0], sep=',',header=6).iloc[-2 * (test_num + per_num):-1 * (test_num + per_num),0:2].values
        print(data_all)
        data_all.dtype = 'float64'

        data = copy.deepcopy(data_all[:-per_num, :])
        y = data_all[-per_num:, :]

        # #归一化
        normalize = np.load("./npy/"+userid+"_traj_model_trueNorm.npy")
        data = self.NormalizeMultUseData(data, normalize)

        model = keras.models.load_model("./model/"+userid+"_traj_model.h5")
        test_X = data.reshape(1, data.shape[0], data.shape[1])
        y_hat = model.predict(test_X)
        y_hat = y_hat.reshape(y_hat.shape[1])
        y_hat = self.reshape_y_hat(y_hat, 2)

        # 反归一化
        y_hat = self.FNormalizeMult(y_hat, normalize)
        print("predict: {0}\ntrue：{1}".format(y_hat, y))
        print('预测均方误差：', self.mse(y_hat, y))
        print('预测直线距离：{:.4f} KM'.format(self.get_distance_hav(y_hat[0, 0], y_hat[0, 1], y[0, 0], y[0, 1])))
        if draw_pic:
            # 画测试样本数据库
            p1 = plt.scatter(data_all[:-per_num, 1], data_all[:-per_num, 0], c='b', marker='o', label='traj_A')
            p2 = plt.scatter(y_hat[:, 1], y_hat[:, 0], c='r', marker='o', label='pre')
            p3 = plt.scatter(y[:, 1], y[:, 0], c='g', marker='o', label='pre_true')
            plt.legend(loc='upper left')
            plt.grid()
            plt.show()

        return  y_hat[:, 0][0],y_hat[:, 1][0]

    # 通过输入轨迹数据进行预测
    def predict_viaInput_and_draw(self,userid, input_data, test_num=6, pred_num=1,draw_pic=True):
        per_num = pred_num
        data_all = input_data
        data_all.dtype = 'float64'

        data = copy.deepcopy(data_all)
        y = data_all[-per_num:, :]

        # #归一化
        normalize = np.load("./npy/"+userid+"_traj_model_trueNorm.npy")
        data = self.NormalizeMultUseData(data, normalize)

        model = keras.models.load_model("./model/"+userid+"_traj_model.h5")
        test_X = data.reshape(1, data.shape[0], data.shape[1])
        y_hat = model.predict(test_X)
        y_hat = y_hat.reshape(y_hat.shape[1])
        y_hat = self.reshape_y_hat(y_hat, 2)

        # 反归一化
        y_hat = self.FNormalizeMult(y_hat, normalize)
        print("predict: {0}\ntrue：{1}".format(y_hat, y))
        #print('预测均方误差：', self.mse(y_hat, y))
        #print('预测直线距离：{:.4f} KM'.format(self.get_distance_hav(y_hat[0, 0], y_hat[0, 1], y[0, 0], y[0, 1])))
        if draw_pic:
            # 画测试样本数据库
            p1 = plt.scatter(data_all[:-per_num, 1], data_all[:-per_num, 0], c='b', marker='o', label='traj_A')
            p2 = plt.scatter(y_hat[:, 1], y_hat[:, 0], c='r', marker='o', label='pre')
            plt.title(userid)
            #p3 = plt.scatter(y[:, 1], y[:, 0], c='g', marker='o', label='pre_true')
            plt.legend(loc='upper left')
            plt.grid()
            plt.show()

        return  y_hat[:, 0][0],y_hat[:, 1][0]

if __name__ == '__main__':
    trained_model = Predict_via_LSTM_model()
    input_data = np.array([[ 40.008945, 116.321551],
                 [ 40.008851, 116.321485],
                 [ 40.008607, 116.321862],
                 [ 40.008652, 116.322251],
                 [ 40.008897, 116.321603],
                 [ 40.008928, 116.32161 ]])
    #x,y = trained_model.predict_fromFile_and_draw(userid='000',test_num=6,pred_num=1)
    x, y = trained_model.predict_viaInput_and_draw(userid='000',input_data=input_data,test_num=6,pred_num=1)
    print(x,y)
