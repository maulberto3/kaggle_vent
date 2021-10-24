from joblib import dump, load
from tqdm import tqdm
from time import perf_counter
from random import choice
from pprint import pprint
#
from numpy import column_stack
from pandas import DataFrame, merge, read_csv
#
from torch import (load as pt_load, set_printoptions, unique, device, cuda,
                   zeros, no_grad, abs as pt_abs, mean, as_tensor, save as
                   pt_save)
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module, LSTM, Linear, CELU, SELU, Dropout
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, lr_scheduler

set_printoptions(precision=2, linewidth=150, sci_mode=False)

device = device("cuda" if cuda.is_available() else "cpu")
print(device)
batch_size = 503
loader_shuffle = False
columns = ['R', 'C', 'time_step', 'u_in', 'u_out', 'pressure']
epochs = 1


def fit_cluster_model_to_train(train_X, n_clusters=1000):
    """Fit train y (pressure) cluster model"""
    train_X, train_y = train_X.numpy()[:, 2:-1], train_X.numpy()[:, -1]
    from sklearn.cluster import MiniBatchKMeans
    try:
        print('Using fitted Cluster model.')
        mk = load('cluster_model.joblib')
    except BaseException as e:
        print('==>Error:', e)
        print('Fitting Cluster Model...')
        mk = MiniBatchKMeans(n_clusters)
        mk.fit(train_X)
        print('Fitted.')
        dump(mk, 'cluster_model.joblib')
    pred = mk.predict(train_X)
    res = column_stack((pred, train_y))
    res = DataFrame(res, columns=['cluster', 'pressure'])
    res = res.pivot_table(index='cluster', values='pressure', aggfunc='mean')
    return mk, res


train_raw = read_csv('/kaggle/input/ventilator-pressure-prediction/train.csv')
train_raw = as_tensor(train_raw.to_numpy()).float()
train_raw.shape

test_raw = read_csv('/kaggle/input/ventilator-pressure-prediction/test.csv')
test_raw = as_tensor(test_raw.to_numpy()).float()
test_raw.shape

cl_model, cl_to_press = fit_cluster_model_to_train(train_raw)
cl_model, cl_to_press

s = test_raw.shape[0]
print(list(div for div in range(256, 1024) if s % (div * 80) == 0))


class DS(Dataset):
    def __init__(self, data, train_data=True):
        self.data = data
        self.idxs = self.data[:, 1]
        self.idxs = unique(self.idxs)
        self.train_data = train_data

    def __len__(self):
        res = int(self.idxs.shape[0])
        return res

    def __getitem__(self, idx):
        rand = choice(self.idxs)
        mask = (self.data[:, 1] == rand)
        sample = self.data[mask, :]
        if self.train_data:
            X = sample[:, 2:-1]
            y = sample[:, -1].unsqueeze(-1)
            return X.float(), (y /
                               10).float()  # pseudo normalize for convergence
        else:
            X = sample[:, 2:]
            return X.float()


train_ds = DS(train_raw)
test_ds = DS(test_raw, train_data=False)
# train_ds = iter(train_ds)
# print(next(train_ds))
# print(next(train_ds))

train_dl = DataLoader(train_ds,
                      batch_size=batch_size,
                      shuffle=loader_shuffle,
                      drop_last=True,
                      num_workers=-1)
test_dl = DataLoader(test_ds,
                     batch_size=batch_size,
                     shuffle=loader_shuffle,
                     drop_last=False,
                     num_workers=-1)
# train_dl = iter(train_dl)
# batch = next(train_dl)
# X_, y_ = batch
# print(X_.shape)
# print(y_.shape)


class NN(Module):
    def __init__(
            self,
            input_size,
            hidden_size=64,
            num_layers=1,
            #  dropout=0.2,
            device=device,
            batch_size=16):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = Dropout(0.1)
        self.device = device
        self.batch_size = batch_size
        self.activ = SELU()

        self.lstm = LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            #  dropout=self.dropout,
            batch_first=True,
            device=self.device)

        self.linear = Linear(self.hidden_size, 1, device=self.device)
        self.linear2 = Linear(1, 1, device=self.device)

    def init_h(self):
        h = zeros(self.num_layers,
                  self.batch_size,
                  self.hidden_size,
                  device=self.device)
        return h

    def init_c(self):
        c = zeros(self.num_layers,
                  self.batch_size,
                  self.hidden_size,
                  device=self.device)
        return c

    def forward(self, x, h, c, y):
        res, (h, c) = self.lstm(x, (h, c))
        res = self.activ(res)
        # TODO add dropout
        res = self.dropout(res)
        res = self.activ(self.linear(res))
        err = res - y
        # TODO add dropout
        res = self.dropout(res)
        res -= self.linear2(err)
        return res

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# net = NN(X_.shape[-1])
# h = net.init_h()
# c = net.init_c()
# out = net(X_.float().cuda(), h.cuda(), c.cuda())
# print(out.shape)

net = NN(5, batch_size=batch_size)
net.train()
print(f'==> Model # weights: {net.count_parameters():,}')

# criterion = mean absolute error
learning_rate = 0.01
optimizer = Adam(net.parameters(), learning_rate)
lambda1 = lambda epoch: 0.999**epoch
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

# Train
for epoch in range(epochs):
    batches_loss = []
    h = net.init_h()
    c = net.init_c()
    step = 0
    for X, y in tqdm(train_dl):
        out = net(X.float().cuda(), h, c, y.cuda())
        h = h.detach()
        c = c.detach()

        loss = pt_abs(out - y.cuda())
        loss = loss.mean()

        optimizer.zero_grad()
        loss.mean().backward()
        clip_grad_norm_(net.parameters(), 1)
        optimizer.step()
        scheduler.step()

        batches_loss.append(loss)
        step += 1

        print(f'Running MAE {as_tensor(batches_loss).mean():.4f}')

    pt_save(net.state_dict(),
            f'trained_model_MAE_{mean(as_tensor(batches_loss)):.2f}.pth')


def predict_cluster_to_y_value(batch_X, cluster_model, cluster_to_pressure,
                               device):
    tens = []
    for batch in batch_X:
        res = cluster_model.predict(batch)
        res = DataFrame(res, columns=['cluster'])
        res = merge(res,
                    cluster_to_pressure,
                    left_on='cluster',
                    right_on='cluster')
        tens.append(res['pressure'].tolist())
    tens = as_tensor(tens)
    tens = tens.reshape(tens.shape[0], 80, 1)
    tens = tens.to(device)
    return tens


# Inference
with no_grad():
    net.train(False)
    subm = []
    step = 0
    for i, X in enumerate(test_dl):
        y = predict_cluster_to_y_value(X.numpy(), cl_model, cl_to_press,
                                       device)
        out = net(X.float().cuda(), h, c,
                  y) * 10  # un-pseudo normalize for inference
        out = out.reshape(out.shape[0] * out.shape[1], 1)
        subm.append(out.cpu().numpy())
        step += 1

# Submission
subm = DataFrame([float(pred) for arr in subm for pred in arr])
subm.index = range(1, subm.shape[0] + 1)
subm = subm.reset_index()
subm.columns = ['id', 'pressure']
subm

subm.to_csv('submission.csv', index=False)
