from LSTM import LSTM_createModel
from LSTMPredict import Predict_via_LSTM_model
import numpy as np
import matplotlib.pyplot as plt


def train_every_LSTM():
    #创建LSTM_createModel对象，用于训练模型
    newmodel = LSTM_createModel(181)
    newmodel.train_every_user()
    print("Every LSTM model has been trained")

def train_spec_LSTM(userid):
    #创建LSTM_createModel对象，用于训练模型
    newmodel = LSTM_createModel(181)
    newmodel.train_and_save(userid=userid,train_num=6,pred_num=1,draw_pic=True)
    print("LSTM model of "+userid+" has been trained")

def predict_via_File(userid):
    x,y = trained_model.predict_fromFile_and_draw(userid=userid,test_num=6,pred_num=1)
    print("Predict via file complete")
    return x, y

def predict_via_Data(userid, data):
    x, y = trained_model.predict_viaInput_and_draw(userid=userid, input_data=data, test_num=6, pred_num=1)
    print("Predict via data complete")
    return x, y

def gain_User_Group(userid):
    usergrp = ['001','002']
    return usergrp

def draw_Final_Pic(input_data, x_final, y_final, userid):
    # 画测试样本数据库
    print(input_data)
    plt.scatter(input_data[:, 1], input_data[:, 0], c='b', marker='o', label='traj_A')
    p2 = plt.scatter(y_final,x_final,  c='r', marker='o', label='pre')
    plt.title(userid+" final")
    plt.legend(loc='upper left')
    plt.autoscale(enable=True,axis="both")
    plt.grid()
    plt.show()

def main(userid='000',fileOrdata='file',input_data=None,):
    # 历史轨迹预测权重
    weight_his = 0.8
    weight_sim = 1 - weight_his
    # 预测前需要将模型训练好
    # 如果已经提前训练好并保存，可以直接进行预测
    # train_every_LSTM()
    # train_spec_LSTM('000')

    # 创建Predict_via_LSTM_model对象，用于预测用户轨迹输出
    trained_model = Predict_via_LSTM_model()
    # 两种预测方式：一种是直接输入用户历史轨迹，一种是在已有文件中获取输入做展示

    # 首先根据历史信息进行轨迹预测
    if fileOrdata == 'file':
        x_history, y_history = predict_via_File(userid)
    else:
        x_history, y_history = predict_via_Data(userid, input_data)

    # 预测经度保存在x，纬度保存在y
    # print(x_history,y_history)

    # 再根据用户相似信息进行轨迹预测
    # 首先获取相似用户列表
    user_grp = gain_User_Group(userid)

    x_sim, y_sim = 0, 0
    for user in user_grp:
        x_tmp, y_tmp = predict_via_Data(user, input_data)
        print(user, ":", x_tmp, y_tmp)
        x_sim += x_tmp
        y_sim += y_tmp
    x_sim = x_sim / len(user_grp)
    y_sim = y_sim / len(user_grp)

    x_final = (x_history * weight_his) + (x_sim * weight_sim)
    y_final = (y_history * weight_his) + (y_sim * weight_sim)
    print("Final result of prediction")
    print("Next Position of user " + userid + " is ({0},{1})".format(x_final, y_final))

    #绘制最终预测图象
    draw_Final_Pic(input_data, x_final, y_final, userid)

if __name__ == '__main__':
    main(userid='000',fileOrdata='data',input_data = np.array([[40.008945, 116.321551],
                           [40.008851, 116.321485],
                           [40.008607, 116.321862],
                           [40.008652, 116.322251],
                           [40.008897, 116.321603],
                           [40.008928, 116.32161]]))