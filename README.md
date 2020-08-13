# Position_Predict_Based_on_LSTM
Using LSTM to predict user position based on history GPS data of the user
使用LSTM网络预测用户的位置
## 数据集
使用的是**Geolife Trajectories 1.3**，当中有181位用户信息，由微软亚洲研究院收集的GPS信息（从2007到2012年）

## 训练模型
训练模型的代码存放在**LSTM.py**

## 预测位置
在训练好模型后调用模型进行预测，代码存放在**LSTMPredict.py**

## 主函数
**main.py**为封装的主函数
