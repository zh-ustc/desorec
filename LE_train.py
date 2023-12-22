from K_means import K_means,m_vector,A_vector
from get_user import getuser
from args import args
import numpy as np
import torch
from graph import graph,save_index

T = {}
T['lastfm_deepfm'] = 0.25
T['lastfm_sasrec'] = 2.0
T['lastfm_mlp'] = 0.5
T['lastfm_GRU4Rec'] = 2.0
T['ml146_deepfm'] = 0.25
T['ml146_sasrec'] = 0.5
T['ml146_mlp'] = 0.25
T['ml146_GRU4Rec'] = 0.5
T['Diginetica_deepfm'] = 0.25
T['Diginetica_sasrec'] = 1.0
T['Diginetica_mlp'] = 1.0
T['Diginetica_GRU4Rec'] = 2.0

with torch.no_grad():

    #getuser(args)
    
    # exit()

    try:
        user = np.load(args.load_path+'user.npy')
        user = torch.tensor(user,device='cuda')
        label= np.load(args.load_path+'target.npy')
        label = torch.tensor(label).to(args.device).reshape(-1)
    except:
        print('begin get user embd.......')
        getuser(args)
        user = np.load(args.load_path+'user.npy')
        user = torch.tensor(user,device='cuda')
        label= np.load(args.load_path+'target.npy')
        label = torch.tensor(label).to(args.device).reshape(-1)
        print('finish get user embd.......')

    print(user.shape)
    



    
    print('begin graph.......')
    try:
        dis = np.load(args.load_path+'dis'+str(args.class_num)+'.npy')
        dis = torch.tensor(dis,device='cuda')
    except:
        print('begin kmeans.......')
        k = K_means(user,args.class_num,args.device)
        dis,m,_,center = k.f()
        np.save(args.load_path+'dis'+str(args.class_num),dis.cpu().numpy())
        print('finish kmeans.......')
    
    #m = torch.min(dis,-1)[0].mean().cpu().item()
    args.tau = T[args.dataname+'_'+args.model]
    path = args.load_path+'class-{}-tau{}'.format(str(args.class_num),str(args.tau))
    graph(dis,user,label,path,args.tau,0.5)
    save_index(dis,path,args.tau)
    print('finish graph.......')


