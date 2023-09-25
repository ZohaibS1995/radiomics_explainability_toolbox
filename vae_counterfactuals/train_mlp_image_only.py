# other imports
import numpy as np
import os
from barbar import Bar

# sklearn imports
from sklearn import metrics

# torch imports
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset


# other class imports
from utils import EarlyStopping
from config import config


def split_train_val(train_data, fold_val):
    t_val_data = []
    t_train_data = []

    for idx in range(5):

        if idx == fold_val:
            t_val_data.extend(train_data[idx])
        else:
            t_train_data.extend(train_data[idx])

    return t_train_data, t_val_data


class ClassificationDataset_only_image(Dataset):

    def __init__(self, value_names, labels):
        self.value_names = value_names
        self.label = labels

    def __len__(self):
        return len(self.value_names)

    def __getitem__(self, idx):
        value = self.value_names[idx]
        label = torch.tensor(self.label[idx])
        label = label.long()
        sample = {'value': value, 'label': label}
        return sample


class MLP_only_image(torch.nn.Module):
    def __init__(self):
        super(MLP_only_image, self).__init__()
        self.fc1 = torch.nn.Linear(512, 256)
        torch.nn.init.zeros_(self.fc1.weight)

        self.fc2 = torch.nn.Linear(256, 128)
        torch.nn.init.zeros_(self.fc2.weight)

        self.fc3 = torch.nn.Linear(128, 32)
        torch.nn.init.zeros_(self.fc3.weight)

        self.fc4 = torch.nn.Linear(32, 2)

    def forward(self, in_values):
        output = F.relu(self.fc1(in_values))
        output = F.relu(self.fc2(output))
        output = F.relu(self.fc3(output))
        output = self.fc4(output)
        return output


if __name__ == "__main__":

    mlp_patience_early_stopping = 20
    mlp_lr = 1e-3
    mlp_weight_decay = 0.1
    mlp_batch_size = 8
    mlp_num_epochs = 200
    mlp_scheduler_patience = 5
    mlp_min_lr = 10e-5

    # training the MLP model
    train_data = np.load("latent_image_dict_train.npy", allow_pickle=True)
    train_label = np.load("latent_label_dict_train.npy", allow_pickle=True)

    train_data = [value for key, value in train_data.tolist().items()]
    train_label = [value for key, value in train_label.tolist().items()]

    val_data = np.load("latent_image_dict_val.npy", allow_pickle=True)
    val_label = np.load("latent_label_dict_val.npy", allow_pickle=True)

    val_data = [value for key, value in val_data.tolist().items()]
    val_label = [value for key, value in val_label.tolist().items()]

    fold_no = 0

    # getting the latent data
    train_latent_space = train_data[fold_no]
    val_latent_space = val_data[fold_no]

    train_labels = train_label[fold_no]
    val_labels = val_label[fold_no]

    # setting training params
    early_stopping = EarlyStopping(patience=10, verbose=True,  path=os.path.join(config.model_save_dir, f"{fold_no}_mlp_checkpoint.pt"))

    training_dataset = ClassificationDataset_only_image(train_latent_space, train_labels)
    training_dataloader = DataLoader(training_dataset, batch_size=16, shuffle=True, num_workers=0)

    validation_dataset = ClassificationDataset_only_image(val_latent_space, val_labels)
    validation_dataloader = DataLoader(validation_dataset, batch_size=16, shuffle=False,
                                       num_workers=0)

    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []

    # instantiate model
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")  ## specify the GPU id's, GPU id's start from 0.
    model = MLP_only_image()
    model = model.to(device)

    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=mlp_lr, weight_decay=mlp_weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=mlp_scheduler_patience, min_lr=mlp_min_lr)
    criterion = nn.CrossEntropyLoss()

    # training loop
    training = True
    epoch = 1

    try:
        while training:

            # epoch specific metrics
            train_loss = 0
            mask_loss = 0
            train_accuracy = 0
            val_loss = 0
            val_accuracy = 0
            total_loss = 0

            proba_t = []
            true_t = []
            proba_v = []
            true_v = []
            # -----------------------------
            # training samples
            # -----------------------------

            # set the model into train mode

            model.train()
            for b, batch in enumerate(Bar(training_dataloader)):

                x = batch['value'].to(device)
                y = batch['label'].to(device)


                # Type Casting
                x = x.type('torch.cuda.FloatTensor')

                # clear gradients
                optimizer.zero_grad()

                # infer the current batch
                preds = model(x)

                # compute the loss.
                loss = criterion(preds, y)
                train_loss += loss.item()

                # backward loss and next step
                loss.backward()
                optimizer.step()

                # compute the accuracy
                pred = preds.max(1, keepdim=True)[1]
                batch_accuracy = pred.eq(y.view_as(pred).long())
                train_accuracy += (batch_accuracy.sum().item() / np.prod(y.shape))

                for i in range(len(y)):
                    proba_t.append(np.squeeze(F.softmax(preds[i]).cpu().detach().numpy()))
                    true_t.append(batch['label'].detach().numpy()[i])

            fpr, tpr, _ = metrics.roc_curve(np.array(true_t), np.array(proba_t)[:, 1])
            train_auc = metrics.auc(fpr, tpr)

            # -----------------------------
            # validation samples
            # -----------------------------

            # set the model into train mode
            model.eval()
            for b, batch in enumerate(Bar(validation_dataloader)):

                x = batch['value'].to(device)
                y = batch['label'].to(device)

                x = x.type('torch.cuda.FloatTensor')

                # infer the current batch
                with torch.no_grad():
                    preds = model(x)
                    loss = criterion(preds, y)
                    val_loss += loss.item()

                    # compute the accuracy
                    pred = preds.max(1, keepdim=True)[1]
                    batch_accuracy = pred.eq(y.view_as(pred).long())
                    val_accuracy += batch_accuracy.sum().item() / np.prod(y.shape)
                    for i in range(len(y)):
                        proba_v.append(np.squeeze(F.softmax(preds[i]).cpu().detach().numpy()))
                        true_v.append(batch['label'].detach().numpy()[i])

            fpr, tpr, _ = metrics.roc_curve(np.array(true_v), np.array(proba_v)[:, 1])
            val_auc = metrics.auc(fpr, tpr)
            scheduler.step(val_loss)

            # compute mean metrics
            train_loss /= (len(training_dataloader))
            train_accuracy /= (len(training_dataloader))
            val_loss /= (len(validation_dataloader))
            val_accuracy /= (len(validation_dataloader))
            early_stopping(val_loss, model)

            train_loss_all.append(train_loss)
            train_acc_all.append(train_accuracy)
            val_loss_all.append(val_loss)
            val_acc_all.append(val_accuracy)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            print(
                'Epoch {:d} train_loss {:.4f} train_acc {:.4f} train_auc {:.4f} val_loss {:.4f} val_acc {:.4f} val_auc {:.4f}'.format(
                    epoch,
                    train_loss,
                    train_accuracy,
                    train_auc,
                    val_loss,
                    val_accuracy,
                    val_auc))

            if epoch >= mlp_num_epochs:
                training = False

            # update epochs
            epoch += 1
        print('********************************************************************************')

    except KeyboardInterrupt:
        pass
