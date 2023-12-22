import matplotlib.pyplot as plt
import torch
import random
import copy
import numpy as np
from sklearn.decomposition import PCA
import json,pickle

#from args import args

def save_index(dis,path,alpha):
    file = path+'index.json'
    f = open(file, 'w')
    _, l = torch.min(dis, -1)
    l1 = l.cpu().tolist()
    T = {}
    index = {}
    for i in range(dis.shape[0]):
        key = l1[i]
        if key not in T:
            T[key] = 0
        index[i] = (key, T[key])
        T[key] += 1
    json.dump(index, f)

def sta(l,k):
    for i in range(k):
        if (torch.sum(l==i)>m):
            m = torch.sum(l==i)
            j = i
    print(m,j)


def graph(dis,user,label,path,beta,alpha):
    
    with torch.no_grad():
        H = {}
        n = np.zeros(dis.shape[1])
        center, l = torch.min(dis, -1)
        for i in l:
            n[i] += 1
        print('max cluster',max(n))
        enhance = []
        for i in range(dis.shape[1]):

            print(i)
            if torch.sum(l==i) == 0 :
                H[i] = []
                enhance.append([])
                continue
            U = user[l == i]
            L = label[l == i]
            A = torch.zeros(U.shape[0], U.shape[0],device='cuda')

            for j in range(U.shape[0]):
                U1 = U - U[j].unsqueeze(0)
                U1 = torch.sum(U1 ** 2, -1).sqrt()  # /c[j]
                U1[j]=60
                A[j] = (U1 / (-beta)).softmax(-1)
                

            if (A.shape[0]==1) :A[0][0]=1
            hash = {}
            l1 = L.cpu().tolist()
            t = 0
            #print(l1)
            for j in range(U.shape[0]):
                if l1[j] not in hash:
                    hash[l1[j]] = t
                    t += 1
            print(j, t)
            d = torch.zeros(U.shape[0], t, device='cuda')
            for j in range(U.shape[0]):
                x = torch.zeros(t,device='cuda')
                x[hash[l1[j]]] = 1
                d[j] = x

            print(d.shape)
            d0 = d
            for j in range(20):
                d = torch.matmul(A, d) * alpha + (1 - alpha) * d0

            m,_ = torch.max(d,-1)
            print(m[0])
            print('max 1',torch.mean(m))
            H[i] = list(hash.keys())

            # np.save('enhancement_data/label/'+str(i),d.cpu().numpy())
            enhance.append(d.cpu().numpy())
            
            d[d0.bool()] = 0
            m,_ = torch.max(d,-1)
            print('max 2',m.mean())

        pickle.dump(enhance,open(path+'soft','wb'))
        # ----------------save hash i -> label_id----------------------------
        with open(path+'hash.json', 'w') as f:
            json.dump(H, f)





if __name__ == '__main__':
    from args import args
    device = args.device
    path = args.save_path
    path1 = args.load_path
    dis = np.load(path1+'dis.npy')
    dis = torch.tensor(dis).to(device)
    print(dis.shape)
    print(dis[0])
    user = np.load(path1+'user128.npy')
    user = torch.tensor(user).to(device)
    label = np.load(path1+'target128.npy')
    # print(label[-2])
    label = torch.tensor(label).to(device)
    print(label.shape)
    label = label.squeeze(1)
    _,l = torch.min(dis,-1)
    graph(dis,user,label,path)
    save_index(dis,path)





