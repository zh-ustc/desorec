

import matplotlib.pyplot as plt
import torch
import random
import copy
import numpy as np
from sklearn.decomposition import PCA
from torch import nn
import torch.nn.functional as F
from torch import nn
class K_means():
    def __init__(self, data, k ,device):
        self.data = data
        self.k = k
        self.device  = device

    def distance(self, p1, p2):
        return torch.sum((p1-p2)**2).sqrt()

    def generate_center(self):
        # 随机初始化聚类中心
        random.seed(2)
        n = self.data.size(0)
        rand_id = random.sample(range(n), self.k)
        center = []
        for id in rand_id:
            center.append(self.data[id])
        center = torch.cat(center).reshape(self.k,-1)
        return center

    def f(self):
        center = self.generate_center()
        n = self.data.size(0)
        label = torch.zeros(n).long().to(self.device)
        #data = self.data.unsqueeze(1).expand(-1,self.k,-1)
        print(torch.sum(self.data))
  
        for _ in range(100):
            old_label = label
            old_center = copy.deepcopy(center)
            d = []
            for j in range(self.k): # 到每个聚类的距离向量 dis 计算
                dis = self.data - old_center[j]
                # x = torch.sum(dis)
                # if x!=x:
                #     print(old_center[j])
                #     exit()
                
                dis = dis ** 2
                dis = torch.sum(dis,-1).sqrt()
                d.append(dis)

            dis = torch.cat(d).reshape(self.k,-1).transpose(0,1)
            m,label = torch.min(dis,-1)



            maxc = 0
            for j in range(self.k):
                cnum = torch.sum(label == j)
                maxc = max(maxc,cnum.item())
                if cnum>0:
                    center[j] = torch.mean(self.data[label == j], dim=0)
            print(_,torch.mean(m),maxc)
            if(torch.sum(label == old_label)==n): break
            #print(center.shape)
        return dis,m,label,center

def m_vector(dis,b):
    d = []
    dis = dis + 1e-10
    for i in range(dis.shape[1]):
        m = dis[:,i]
        dis1 = dis / (m.reshape(-1,1))
        dis1 = dis1 ** (-1/(b-1))
        x = torch.sum(dis1,-1)
        # y = torch.sum(x)
        # if y!=y:
        #     print('nan')
        #     exit()
        d.append(x)

    m = torch.cat(d).reshape(dis.shape[1],dis.shape[0]).transpose(0,1).cpu().numpy()

    #m = 1/m
    # print(torch.sum(m[:32],-1))
    # _,z = torch.max(m[:32],-1)
    # print(z)
    #np.save('kmeans/m'+str(m.shape[1]),m)
    return 1/m
    #
def A_vector(m,item_num,label):
    label = label.cpu().numpy()
    A = torch.zeros(item_num+1,m.shape[1]).to(m.device)
    S = {}
    for i in range(m.shape[0]):
        if label[i] not in S:
            S[label[i]] = []
        S[label[i]].append(i)

    for j in range(1,item_num+1):
        if j in S:
            x = torch.LongTensor(S[j]).to(m.device)
            x = torch.index_select(m,0,x)
            A[j] = torch.mean(x,dim = 0)

        if j % 10000 ==0 :print(j)
        # if torch.sum(label==j) > 0:
        #     A[j] = torch.mean(m[label == j],dim = 0)
    #np.save('kmeans/A'+str(A.shape[1]),A.cpu().numpy())
    #print(A[1],A[-1])
    return A

def pca_scatter(m,A,label,target):
    m  = m[label==target]
    score = torch.matmul(m[:256],A.transpose(0,1))
    score = score.cpu().numpy()
    pca = PCA(n_components=2)
    X = score
    print(X.shape)
    results = pca.fit_transform(X)
    print(results.shape)
    plt.scatter(results[:,0],results[:,1])

def pca_kmeans(user,label,target):
    user = torch.tensor(user).to(device)
    d = []
    for i in range(3):
        d.append(user[label==target+i])
        #d.append(user[target+i])
    user = torch.cat(d).reshape(-1,user.shape[1])
    user = user.cpu().numpy()
    pca = PCA(n_components=2)
    print(len(user))
    results = pca.fit_transform(user)
    print(results.shape)
    end = 0

    for i in range (3):
        begin = end
        end += d[i].shape[0]
        plt.scatter(results[begin:end, 0], results[begin:end, 1])

if __name__ == '__main__':
    from args import args
    path =  args.load_path
    user = np.load(path+'user128.npy')
    label = np.load(path+'target128.npy')
    print(label[0],label[1])
    device = args.device
    label = torch.tensor(label).to(device)
    #-------------------------------------------kmeans data -> dis
    data = torch.tensor(user).to(device)
    print(data[0])
    model = K_means(data,args.class_num,device)
    dis,m,label,center = model.f()
    np.save(args.save_path+'dis',dis)
