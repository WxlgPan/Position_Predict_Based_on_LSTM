import numpy as np
import os
import random
'''
1.提取坐标信息
'''
class Group_Rec:
    def __init__(self, UserNume):
        self.lat, self.long, self.time = [], [], []
        self.count = []
        self.UserNum = UserNume
    # 计算并生成相似成员信息
    def calSim(self, show_result=False):
        # reinitialize
        self.lat, self.long, self.time = [], [], []
        self.count = []

        # Algorithm begin
        for userid in range(self.UserNum):
            useridstr = str(userid).zfill(3)
            for i, j, filenamelst in os.walk('./Geolife Trajectories 1.3/Data/' + useridstr + '/Trajectory/'):
                for filename in filenamelst:
                    with open(str('./Geolife Trajectories 1.3/Data/'+useridstr+'/Trajectory/'+filenamelst[0]), 'r') as file:
                        data = file.readlines()[6:]  # 从第7行开始读取
                        lat_tmp, long_tmp, time_tmp = [], [], []
                        for line in data:
                            a = line.split(',')
                            lat_tmp.append(float(a[0]))  # 存放纬度
                            long_tmp.append(float(a[1]))  # 存放经度
                            #time_tmp.append(float(a[6]))  # 存放时间
                            line = file.readlines()
                        file.close()
                        self.lat.append(lat_tmp)
                        self.long.append(long_tmp)
                        #self.time.append(time_tmp)
                        self.count.append(len(self.lat[userid]))

        #检验提取的信息

        print(self.count)  #显示不同用户的轨迹长度

        #minCount = min(count)

        for userid1 in range(self.UserNum):
            for userid2 in range(userid1 + 1, self.UserNum):

                minCount = min(len(self.lat[userid1]), len(self.lat[userid2]))  # 取较短的数列长度作为计算长度

                dist_lat = np.linalg.norm(self.lat[userid1][:minCount], self.lat[userid2][:minCount])
                dist_long = np.linalog.norm(self.long[userid1][:minCount], self.long[userid2][:minCount])
                if dist_lat < 1:
                    sim_lat = 1
                else:
                    sim_lat = 0
                if dist_long < 1:
                    sim_long = 1
                else:
                    sim_long = 0
                matrix_dist = np.zeros(shape=(self.UserNum, self.UserNum))  # 创经纬度距离矩阵
                matrix_dist[userid1][userid2] = dist_lat  # 经度距离放入上三角
                matrix_dist[userid2][userid1] = dist_long  # 纬度距离放入下三角
                matrix_result = np.zeros(shape=(self.UserNum, self.UserNum))
                matrix_result[userid1][userid2] = sim_lat  # 将是否相似的结果导入矩阵
                matrix_result[userid2][userid1] = sim_long
        if show_result:
            print(matrix_dist)
            print(matrix_result)
        group_file = open('group.txt', 'w')
        group_file.write(matrix_result)
        group_file.close()
    # 根据已经生成的相似成员信息获取相似成员列表
    def getSimUser(self, userid):
        if not os.path.exists('group.txt'):
            self.calSim()

        matrix_group = []
        with open('group.txt','r') as group_file:
            data = group_file.readlines()

            for line in data:
                data_list = line.split(' ')
                for i in range(len(data_list)):
                    data_list[i] = int(data_list[i])
                matrix_group.append(data_list)
        if userid <= len(matrix_group):
            return matrix_group[userid]
        else:
            print('SimUser has not been calculated')
            return []
    # 根据已经生成的相似成员信息获取相似成员列表tmp
    def getSimUserX(self, userid):
        return ['001','002']

if __name__ == '__main__':
    obj = Group_Rec(UserNume=5)
    obj.calSim(True)














