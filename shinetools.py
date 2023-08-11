import torch
import torch.nn as nn
import torch.utils.data as datasets
import torch.optim as optim
import numpy as np
import torchmetrics as tm
class TDataset(datasets.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

        assert (x.shape[0] == y.shape[0])

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def torch_form(x):
    x = torch.tensor(x, dtype=torch.float32)
    return x

def get_dataloader(x, y, batch_size):
    x = torch_form(x)
    y = torch_form(y)

    dataset = TDataset(x, y)
    return datasets.DataLoader(dataset, batch_size)

class ExposedWeightsModel(nn.Module):
    def __init__(self, dims):
        super().__init__()

        self.dims = dims
        # self.seq = nn.Sequential(
        #     nn.Linear(dims, dims // 30),
        #     nn.ReLU(),
        #     nn.Linear(dims // 30, dims // 100),
        #     nn.ReLU(),
        #     nn.Linear(dims // 100, dims),
        # )

        self.seq = nn.Sequential(
            nn.Linear(dims, dims // 30),
            nn.ReLU(),
            nn.Linear(dims // 30, dims // 10),
            nn.ReLU(),
            nn.Linear(dims // 10, dims),
            nn.Tanh()
        )

        self.adv_net = nn.Sequential(
            nn.Linear(dims, dims),
            nn.Tanh()
        )

        self.optimizer = None
        self.reg_const = None
        self.reg_type = None
        self.adv_const = None
        self.adv_optimizer = None

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def compile(self, optimizer, lr, reg_type, reg_const, adv_const):
        if reg_type not in ["l1", "l2"]:
            raise Exception(f"regularization type {reg_type} is not implemented")

        self.optimizer = optimizer(self.seq.parameters(), lr=lr)
        self.adv_optimizer = optimizer(self.adv_net.parameters(), lr=lr)
        self.reg_const = reg_const  # alpha: regularization constant on loss
        self.reg_type = reg_type
        self.adv_const = adv_const

    def fit(self, dataloader: datasets.DataLoader, epochs=1, verbose=False, validation=None):
        assert self.optimizer is not None
        assert self.reg_const is not None
        assert self.reg_type is not None
        assert self.adv_const is not None
        assert self.adv_optimizer is not None

        for epoch in range(epochs):
            sum_loss = 0
            num_correct = 0
            num_attempt = 0

            auc = tm.AUROC("binary")

            for i, (x, y) in enumerate(dataloader):
                self.optimizer.zero_grad()
                self.adv_optimizer.zero_grad()

                pred_y, weights = self.forward(x)
                classification_loss = self.bce_loss(pred_y, y)
                regularization_loss = torch.sum(torch.abs(weights), dim=1) if self.reg_type == "l1" \
                                            else torch.sum(torch.square(weights), dim=1)
                adv_loss = torch.mean(torch.std(weights, dim=1))
               # adv_loss = self.mse(self.adv_net(x), weights.detach())
                #adv_loss.backward()

                #adv_loss = -self.mse(self.adv_net(x).detach(), weights)

                loss = classification_loss + self.reg_const * torch.mean(regularization_loss) - \
                       self.adv_const * adv_loss
                loss.backward()

                sum_loss += loss.item()
                py = torch.sigmoid(pred_y)
                num_correct += np.sum((torch.round(py).detach().numpy() == y.numpy()))
                num_attempt += x.shape[0]
                self.optimizer.step()
                self.adv_optimizer.step()

                auc.update(py, y)

                if verbose:
                    print("\r", f"epoch {epoch+1}/{epochs}, iter {i+1}/{len(dataloader)}, "
                                f"avg_loss: {sum_loss/(i+1)} acc: {np.round(num_correct/num_attempt, 4)}, "
                                f"auc: {auc.compute()}", end="")

            if validation is None:
                print()
                continue

            test_sum_loss = 0
            test_num_correct = 0
            test_num_attempt = 0

            test_auc = tm.AUROC("binary")

            for i, (x, y) in enumerate(validation):
                with torch.no_grad():
                    pred_y, weights = self.forward(x)
                    classification_loss = self.bce_loss(pred_y, y)
                    regularization_loss = torch.sum(torch.abs(weights), dim=1) if self.reg_type == "l1" \
                                                else torch.sum(torch.square(weights), dim=1)
                    adv_loss = torch.mean(torch.std(weights, dim=1))

                    loss = classification_loss + self.reg_const * torch.mean(regularization_loss) \
                           + self.adv_const * adv_loss

                    test_sum_loss += loss.item()
                    py = torch.sigmoid(pred_y)
                    test_num_correct += np.sum((torch.round(py).detach().numpy() == y.numpy()))
                    test_num_attempt += y.shape[0]
                    test_auc.update(py, y)

                print("\r", f"epoch {epoch+1}/{epochs}, iter {i+1}/{len(validation)}, "
                            f"avg_loss: {sum_loss/(i+1)} acc: {np.round(num_correct/num_attempt, 4)}, "
                            f"auc: {auc.compute()}, test_loss: {test_sum_loss/(i+1)}, "
                            f"test_acc: {np.round(test_num_correct/test_num_attempt, 4)}, "
                            f"test_auc: {test_auc.compute()}", end="")
            print()


    def predict(self, x):
        return self.forward(x)[0]

    def forward(self, x):
        weights = self.seq(x)

        return (weights * x).sum(dim=1), weights
