import time
import torch
from torch._C import device
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from loss import CE

def neg_sample(seq, labels, num_item,sample_size):
    negs = set()
    seen = set(labels)

    while len(negs) < sample_size:
        candidate = np.random.randint(0, num_item) + 1
        while candidate in seen or candidate in negs:
            candidate = np.random.randint(0, num_item) + 1
        negs.add(candidate)
    return negs

class Trainer():
    def __init__(self, args, model, train_loader, test_loader,valid_loader):
        self.args = args
        self.lamda = args.lamda
        self.device = args.device
        self.clean = args.clean
        print(self.device)
        self.model = model.cuda()

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.valid_loader = valid_loader
        self.lr_decay = args.lr_decay_rate
        self.lr_decay_steps = args.lr_decay_steps
        self.target = args.le_share
        self.soft_target = args.soft_target

        self.enable_sample = args.enable_sample
        self.sampled_evaluation = args.sampled_evaluation

        self.cr = CE(self.model, args)
        self.num_epoch = args.num_epoch
        self.metric_ks = args.metric_ks
        self.eval_per_steps = args.eval_per_steps
        self.save_path = args.save_path


        self.step = 0
        self.metric = args.best_metric
        self.best_metric = -1e9
        self.le_share = args.le_share


    def train(self):
        # BERT training
        self.sample_time = 0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: self.lr_decay ** step, verbose=True)
        for epoch in range(self.num_epoch):
            loss_epoch, time_cost = self._train_one_epoch()
            self.result_file = open(self.save_path + '/result.txt', 'a+')
            print('train epoch:{0},loss:{1},training_time:{2}'.format(epoch + 1, loss_epoch, time_cost))
            print('train epoch:{0},loss:{1},training_time:{2}'.format(epoch + 1, loss_epoch, time_cost),
                  file=self.result_file)
            self.result_file.close()
            if self.stop == self.args.early_stop:
                break
        self.result_file = open(self.save_path + '/result.txt', 'a+')
        print(self.best_test)
        self.result_file.close()


    def _train_one_epoch(self):
        t0 = time.perf_counter()
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader)

        loss_sum = 0
        for idx, batch in enumerate(tqdm_dataloader):
            batch = [x.to(self.device) for x in batch]
            
            # with autocast():


            if self.args.loss == 'desorec':
                loss = self.cr.compute_enhance(batch)

            elif self.args.loss == 'ce':
                loss = self.cr.compute(batch)


            loss_sum += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.step += 1
            tqdm_dataloader.set_postfix({'L':loss_sum/(idx+1)})
            #if idx % 100 == 0 : print(loss_sum/(idx+1))
            # if self.step % self.lr_decay_steps == 0:
            #     self.scheduler.step()
            if self.step % self.eval_per_steps == 0:
                self.sample_time = 0
                metric = {}
                for mode in ['test','valid']:
                    metric[mode] = self.eval_model(mode)
                print(metric)
                self.result_file = open(self.save_path + '/result.txt', 'a+')
                print('step{0}'.format(self.step), file=self.result_file)
                print(metric, file=self.result_file)
                self.result_file.close()
                if metric['valid'][self.metric] > self.best_metric:
                    torch.save(self.model.state_dict(), self.save_path + '/model.pkl')
                    self.result_file = open(self.save_path + '/result.txt', 'a+')
                    print('saving model of step{0}'.format(self.step), file=self.result_file)
                    self.result_file.close()
                    self.best_metric = metric['valid'][self.metric]
                    self.best_test = metric['test']
                    self.stop = 0
                else : self.stop += 1

                self.model.train()

        return loss_sum / idx, time.perf_counter() - t0

    def eval_model(self, mode):
        self.model.eval()
        if mode == 'test':
            tqdm_data_loader =  tqdm(self.test_loader)
        else:
            tqdm_data_loader =  tqdm(self.valid_loader)
        metrics = {}


        with torch.no_grad():
            appear10 = torch.zeros(1, self.args.num_item + 1,device='cuda')
            appear20 = torch.zeros(1, self.args.num_item + 1,device='cuda')
            for idx, eval_batch in enumerate(tqdm_data_loader):
                eval_batch = [x.to(self.device) for x in eval_batch]
                metrics_batch = self.compute_metrics(eval_batch)
                
                for k, v in metrics_batch.items():
                    if not metrics.__contains__(k):
                        metrics[k] = v
                    else:
                        metrics[k] += v

            for k, v in metrics.items():
                metrics[k] = v / (idx+1)


        return metrics

    def recalls_and_ndcgs_for_ks(self, scores, labels, ks):  # MRR ，HR和Recall的区别
        metrics = {}

        answer_count = labels.sum(1)
        labels_float = labels.float()
        rank = (-scores).argsort(dim=1)
        cut = rank

        for k in sorted(ks, reverse=True):
            cut = cut[:, :k]
            hits = labels_float.gather(1, cut)
            metrics['Recall@%d' % k] = \
                (hits.sum(1) / torch.min(torch.Tensor([k]).cuda(),
                                         labels.sum(1).float())).mean().cpu().item()

            position = torch.arange(2, 2 + k,device='cuda')
            weights = 1 / torch.log2(position.float())
            dcg = (hits * weights).sum(1)
            idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in answer_count]).cuda()
            ndcg = (dcg / idcg).mean()
            metrics['NDCG@%d' % k] = ndcg.cpu().item()
        return metrics

   
    
    def compute_metrics(self, batch):
        seqs , answers = batch
        scores = self.model(seqs)
        label = answers.view(-1)

        labels = torch.zeros(seqs.shape[0], self.args.num_item + 1,device='cuda')
        row = []
        col = []
        seqs = seqs.tolist()
        answers = answers.tolist()
        for i in range(len(answers)):
            seq = list(set(seqs[i] + answers[i]))
            seq.remove(answers[i][0])
            row += [i] * len(seq)
            col += seq
        scores[row, col] = -1e9


        index_x = torch.arange(len(label),device='cuda')
        labels[index_x,label] = 1
        metrics = self.recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        return metrics

    