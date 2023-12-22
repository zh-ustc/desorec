from dataset import TrainDataset, EvalDataset
from process import Trainer
#from args import args
import torch.utils.data as Data
from tqdm import tqdm
import torch, os
import numpy as np

from model.GRU4rec import GRU4Rec


def getuser(args):

    if args.model == 'GRU4Rec':
        model = GRU4Rec(args)


    train_dataset = TrainDataset(args)
    train_loader = Data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False)
    path = args.load_path
    #
    # test_dataset = EvalDataset(args, mode='test')
    # test_loader = Data.DataLoader(test_dataset, batch_size=args.test_batch_size)
    print('dataset initial ends')
    tqdm_dataloader = tqdm(train_loader)
    model = model.to(args.device)
    model.load_state_dict(torch.load(os.path.join(path+'model.pkl'),map_location=args.device))
    #model.eval()
    #print(args.device)
    m = model.to(args.device)
    m.eval()
    #trainer = Trainer(args, model, train_loader, train_loader, test_loader, cl_loader)
    # r = trainer.eval_model('val')
    # print(r)
    user = []
    target = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm_dataloader):
            batch = [x.to(m.device) for x in batch]
            x,_, label = batch
            target.append(label)
            x = model(x)
            user.append(x)
    target = torch.cat(target)
    print(user[0].shape)
    user = torch.cat(user)

    print(user.shape,target.shape)
    np.save(args.load_path+'user', user.cpu().numpy())
    np.save(args.load_path+'target',target.cpu().numpy())
# user = np.load(args.save_path+'user128.npy')
# target = np.load(args.save_path+'target128.npy')
# print(user.size,target.size)
if __name__ == '__main__':
    from args import args
    getuser(args)


