from LSTM import LSTM_createModel
from LSTMPredict import Predict_via_LSTM_model
import numpy as np

def Train_every_LSTM():
    #创建LSTM_createModel对象，用于训练模型
    newmodel = LSTM_createModel(181)
    newmodel.train_every_user()
    print("Every LSTM model has been trained")

def Train_spec_LSTM(userid):
    #创建LSTM_createModel对象，用于训练模型
    newmodel = LSTM_createModel(181)
    newmodel.train_and_save(userid=userid,train_num=6,pred_num=1,draw_pic=True)
    print("LSTM model of "+userid+" has been trained")

def Predict_via_File(userid):
    x,y = trained_model.predict_fromFile_and_draw(userid=userid,test_num=6,pred_num=1)
    print("Predict via file complete")
    return x, y

def Predict_via_Data(userid, data):
    x, y = trained_model.predict_viaInput_and_draw(userid=userid, input_data=data, test_num=6, pred_num=1)
    print("Predict via data complete")
    return x, y

if __name__ == '__main__':
    #预测前需要将模型训练好
    #如果已经提前训练好并保存，可以直接进行预测
    #Train_every_LSTM()
    #Train_spec_LSTM('000')

    #创建Predict_via_LSTM_model对象，用于预测用户轨迹输出
    trained_model = Predict_via_LSTM_model()
    #两种预测方式：一种是直接输入用户历史轨迹，一种是在已有文件中获取输入做展示
    input_data = np.array([[ 40.008945, 116.321551],
                 [ 40.008851, 116.321485],
                 [ 40.008607, 116.321862],
                 [ 40.008652, 116.322251],
                 [ 40.008897, 116.321603],
                 [ 40.008928, 116.32161 ]])
    x, y = Predict_via_Data('000',input_data)
    # x, y = Predict_via_File('000')

    #预测经度保存在x，纬度保存在y
    print(x,y)