import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

def flat_accuracy_tokens(logits, labels):
    ## compute training accuracy
    labels = torch.tensor(labels)
    preds = torch.tensor(logits)
    labels = labels.transpose(0, 1)
    preds = preds.transpose(0, 1)
    un_pad_mask = labels != -100
    # print("Enter Accuracy method!!")
    # print("labels shape before pad remove::", labels.shape)
    # print("preds shape before pad remove::", preds.shape)
    labels_unpad = labels[un_pad_mask]
    preds_unpad = preds[un_pad_mask]
    # print("labels shape after pad remove::", labels_unpad.shape)
    # print("preds shape after pad remove::", preds_unpad.shape)
    labels_unpad = labels_unpad.reshape(labels.size(0), -1).t()
    preds_unpad = preds_unpad.reshape(preds.size(0), -1).t()
    # print("labels shape after reshaping to axis=0::", labels_unpad.shape)
    # print("preds shape after reshaping to axis=0::", preds_unpad.shape)
    preds_unpad = torch.argmax(preds_unpad,dim=1)
    labels_unpad = torch.argmax(labels_unpad,dim=1)
    # print("accuracy function inputs:: ", preds_unpad.shape, labels_unpad.shape)
    # print("LABELS HERE:::")
    # print(list(labels_unpad.numpy()))
    # print("PREDICTIONS HERE:::")
    # print(list(preds_unpad.numpy()))

    # print(preds_unpad.shape,labels_unpad.shape)
    # print(preds_unpad)
    # print(labels_unpad)
    # print("/n/n")
    accuracy = accuracy_score(labels_unpad.numpy(), preds_unpad.numpy())
    argmax_mask1 = labels_unpad != 0
    argmax_mask2 = preds_unpad != 0
    preds_unpad1 = preds_unpad[argmax_mask1]
    labels_unpad1 = labels_unpad[argmax_mask1]
    preds_unpad2 = preds_unpad[argmax_mask2]
    labels_unpad2 = labels_unpad[argmax_mask2]
    # print("End of Accuracy method!!\n\n")
    if labels_unpad1.shape[0] == 0 or labels_unpad2.shape[0] == 0:
        return 0.0, 0.0, 0.0, 0
    tpr_scr = accuracy_score(labels_unpad1.numpy(), preds_unpad1.numpy())
    recall_scr = accuracy_score(labels_unpad2.numpy(), preds_unpad2.numpy())
    return accuracy, tpr_scr, recall_scr, 1

def train_fn(data_loader, model, optimizer, device, scheduler, opt):
    model.train()
    # Reset the total loss for this epoch.
    final_loss = 0
    lens = 0
    total_accuracy = 0
    loss_vals = []
    total_tpr = 0
    total_recall = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        data_keyed = {}
        for k, v in zip(["ids", "mask", "context_vec", "target_tag"],data):
            data_keyed[k] = v
        for k, v in data_keyed.items():
            data_keyed[k] = v.to(device)
        optimizer.zero_grad()
        data_keyed["opt"] = opt
        labels, loss = model(**data_keyed)
        loss_vals.append(loss)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()
        # Move loss, labels and input ids to CPU
        loss = loss.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        data_keyed.pop("opt")
        for k, v in data_keyed.items():
            data_keyed[k] = v.to('cpu').numpy()
        accuracy, tpr, recall, consider = flat_accuracy_tokens(labels, data_keyed["target_tag"])
        total_accuracy += accuracy
        total_tpr += tpr
        total_recall += recall
        lens += consider
    torch.cuda.empty_cache()
    # print(loss_vals)
    # print('Memory Allocated in GPU(5):', round(torch.cuda.memory_allocated(5) / 1024 ** 3, 1), 'GB')
    # print('Memory Allocated in GPU(6):', round(torch.cuda.memory_allocated(6) / 1024 ** 3, 1), 'GB')
    # print('Memory Allocated in GPU(7):', round(torch.cuda.memory_allocated(7) / 1024 ** 3, 1), 'GB')
    return final_loss/len(data_loader), total_accuracy/len(data_loader), total_tpr/len(data_loader), total_recall/len(data_loader)


def eval_fn(data_loader, model, device, opt):
    model.eval()
    # Reset the total loss for this epoch.
    final_loss = 0
    lens = 0
    total_accuracy = 0
    total_tpr = 0
    total_recall = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        data_keyed = {}
        for k, v in zip(["ids", "mask", "context_vec", "target_tag"], data):
            data_keyed[k] = v
        for k, v in data_keyed.items():
            data_keyed[k] = v.to(device)
        data_keyed["opt"] = opt
        labels, loss = model(**data_keyed)
        loss = loss.mean()
        final_loss += loss.item()
        # Move loss, labels and input ids to CPU
        loss = loss.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        data_keyed.pop("opt")
        for k, v in data_keyed.items():
            data_keyed[k] = v.to('cpu').numpy()
        accuracy, tpr, recall, consider = flat_accuracy_tokens(labels, data_keyed["target_tag"])
        total_accuracy += accuracy
        total_tpr += tpr
        total_recall += recall
        lens += consider
    torch.cuda.empty_cache()
    # print('Memory Allocated in GPU(5):', round(torch.cuda.memory_allocated(5) / 1024 ** 3, 1), 'GB')
    # print('Memory Allocated in GPU(6):', round(torch.cuda.memory_allocated(6) / 1024 ** 3, 1), 'GB')
    # print('Memory Allocated in GPU(7):', round(torch.cuda.memory_allocated(7) / 1024 ** 3, 1), 'GB')
    return final_loss/len(data_loader), total_accuracy/len(data_loader), total_tpr/len(data_loader), total_recall/len(data_loader)

def test_fn(data_loader, model, device, opt):
    model.eval()
    total_test_accuracy = 0
    labels = []
    lens = 0
    total_tpr = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        data_keyed = {}
        # data_keyed["opt"] = opt
        for k, v in zip(["ids", "mask", "context_vec"], data):
            data_keyed[k] = v
        data_keyed["target_tag"] = torch.tensor([])
        for k, v in data_keyed.items():
            data_keyed[k] = v.to(device)
        data_keyed["opt"] = opt
        label = model(**data_keyed)
        label = label.detach().cpu().numpy()
        labels.extend(label)
        data_keyed.pop("opt")
        for k, v in data_keyed.items():
            data_keyed[k] = v.to('cpu').numpy()
        # accuracy, tpr, f1, consider = flat_accuracy_tokens(label, data_keyed["target_tag"])
        # total_test_accuracy += accuracy
        # total_tpr += tpr
        # lens += consider
    torch.cuda.empty_cache()
    # print("TOTAL TEST ACCURACY:: ",total_test_accuracy/lens)
    # print("TOTAL TEST TPR:: ", total_tpr/lens)
    return np.array(labels)