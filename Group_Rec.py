import numpy as np
import random
import os

'''
1.提取坐标信息
'''''

class Group_Rec:
    def __init__(self, UserNume):
        self.lat, self.long, self.time = [], [], []
        self.count = []
        self.UserNum = UserNume
    def calSim(self, show_result=False):
        # reinitialize
        self.lat, self.long, self.time = [], [], []
        self.count = []

# Algorithm begin
        for userid in range(self.UserNum):
            useridstr = str(userid).zfill(3)
            for dirpath, dirnames,  filenames in os.walk('./Geolife Trajectories 1.3/Data/' + useridstr + '/Trajectory/'):

                    for filename in filenames:
                        with open('./Geolife Trajectories 1.3/Data/'+ useridstr +'/Trajectory/' + filename , 'r') as file:
                            data = file.readlines()[6:]  # 从第7行开始读取
                            lat_tmp, long_tmp, time_tmp = [], [], []
                            for line in data:
                                a = line.split(',')
                                lat_tmp.append(float(a[0]))     # 存放纬度
                                long_tmp.append(float(a[1]))    # 存放经度
                                time_tmp.append((a[6]))    # 存放时间
                                line = file.readlines()
                            file.close()
                            self.lat.append(lat_tmp)
                            self.long.append(long_tmp)
                            self.time.append(time_tmp)
                            self.count.append(len(self.lat[userid]))

        #检验提取的信息

        print(self.count)  #显示不同用户的轨迹长度



        #minCount = min(count)
        matrix_dist = np.zeros(shape=(self.UserNum, self.UserNum))  # 创经纬度距离矩阵
        matrix_result = np.zeros(shape=(self.UserNum, self.UserNum), dtype=int)
        for userid1 in range(self.UserNum):
            for userid2 in range(userid1+1, self.UserNum):

                minCount = min(len(self.lat[userid1]), len(self.lat[userid2]))      #取较短的数列长度作为计算长度

                vec1 = np.array(self.lat[userid1][:minCount])
                vec2 = np.array(self.lat[userid2][:minCount])
                dist_lat = np.linalg.norm(vec1-vec2)
                matrix_dist[userid1][userid2] = dist_lat  # 经度距离放入上三角
                if dist_lat < 0.1:
                    sim_lat = 1
                else:
                    sim_lat = 0
                matrix_result[userid1][userid2] = sim_lat  # 将是否相似的结果导入矩阵
                vec3 = np.array(self.long[userid1][:minCount])
                vec4 = np.array(self.long[userid2][:minCount])
                dist_long = np.sqrt(np.sum(np.square(vec3 - vec4)))
                matrix_dist[userid2][userid1] = dist_long  # 纬度距离放入下三角
                if dist_long < 0.1:
                    sim_long = 1
                else:
                    sim_long = 0
                matrix_result[userid2][userid1] = sim_long



        if show_result:
            print(matrix_dist)
            print(matrix_result)
        sim = 0
        for i in range(self.UserNum):
            a = matrix_result[0][i]
            b = matrix_result[i][0]
            if a==b==1:
                sim = sim+1
        print(sim)          #与用户1相似的用户数
        ARRS = []
        f = open('group.txt', 'w+')
        for i in range(self.UserNum):
            jointsFrame = matrix_result[i]  # 每行
            ARRS.append(jointsFrame)
            for Ji in range(self.UserNum):   #每列
                strNum = str(jointsFrame[Ji])
                f.write(strNum)
                f.write(' ')
            f.write('\n')
        f.close()

        def getSimUser(self, userid):
            if not os.path.exists('group.txt'):
                self.calSim()

            matrix_group = []
            result_lst = []
            with open('group.txt', 'r') as group_file:
                data = group_file.readlines()

                for line in data:
                    data_list = line.split(' ')
                    for i in range(len(data_list)):
                        data_list[i] = int(data_list[i])
                    matrix_group.append(data_list)

            if userid <= len(matrix_group):
                for i in range(len(matrix_group)):
                    if matrix_group[userid][i] == 1:
                        result_lst.append(str(i).zfill(3))
                return result_lst
            else:
                print('SimUser has not been calculated')
                return []

        # 根据已经生成的相似成员信息获取相似成员列表tmp
        def getSimUserX(self, userid):
            return ['001', '002']

if __name__ == '__main__':
    obj = Group_Rec(UserNume=30)
    obj.calSim(True)















