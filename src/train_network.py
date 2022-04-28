import numpy as np
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import time
import os
import interpretability as interpretability

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
# can add the path inside Summary Writer
writer = SummaryWriter()

def get_num_correct(output, labels):
    output = output.cpu().detach()
    labels = labels.cpu().detach()

    output_softmax = nn.functional.softmax(output, dim = 1)
    pred = torch.argmax(output_softmax, dim=1)
    return np.sum(np.array(pred) == np.array(labels))

def train_epoch(
        epoch,
        model,
        train_loader,
        optimizer,
        log_tensorboard
        ):
    losses = []
    total_correct = 0
    for iter, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        # send inputs and labels to the same device
        inputs = inputs.type(torch.FloatTensor).to(DEVICE)
        labels = labels.type(torch.FloatTensor).to(DEVICE)

        outputs = model(inputs)

        outputs = outputs.type(torch.FloatTensor).to(DEVICE)
        labels = labels.type(torch.LongTensor).to(DEVICE)

        loss = criterion(outputs, labels)
        loss_val = loss.detach().item()

        losses.append(loss_val)
        total_correct += get_num_correct(outputs.detach(), labels.detach())

        # if log_tensorboard and iter % 250 == 0:
        #     writer.add_scalar("Batch Loss", loss_val, iter)

        #backprop
        loss.backward()

        # update weights
        optimizer.step()

        del outputs
        del labels
        del inputs

        if iter % 250 == 0:
            print("epoch: {}, iter: {}, loss: {}".format(epoch, iter, loss_val))
    return losses, total_correct

def train(
        model,
        train_loader,
        idx,
        model_hyperparams,
        log_tensorboard=False
        ):
    model.to(DEVICE)
    dataset_length = len(train_loader.dataset)
    lr = model_hyperparams['lr']
    lr_decay = model_hyperparams['lr_decay']
    epochs = model_hyperparams['epochs']

    optimizer = torch.optim.Adagrad(
            model.parameters(),
            lr=lr,
            lr_decay=lr_decay
           )

    train_losses = []
    train_accuracies = []
    for epoch in range(epochs):
        model.train()
        ts = time.time()
        losses, total_correct = train_epoch(epoch, model, train_loader, optimizer, log_tensorboard)

        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        average_loss = np.mean(np.array(losses))
        accuracy = total_correct/dataset_length

        if log_tensorboard:
            writer.add_scalar("Loss/train", average_loss, epoch)
            writer.add_scalar("Correct", total_correct, epoch)
            writer.add_scalar("Accuracy", accuracy, epoch)

        train_losses.append(average_loss)
        train_accuracies.append(accuracy)

        writer.flush()
        writer.close()

    return model, train_losses, train_accuracies

def train_attack_model(
        model,
        train_loader,
        dataset_length,
        attack_hyperparams,
        log_tensorboard=False
        ):
    model.to(DEVICE)
    lr = attack_hyperparams['lr']
    lr_decay = attack_hyperparams['lr_decay']
    epochs = attack_hyperparams['epochs']

    optimizer = torch.optim.Adagrad(
            model.parameters(),
            lr=lr,
            lr_decay=lr_decay
           )

    train_losses = []
    train_accuracies = []
    for epoch in range(epochs):
        model.train()
        ts = time.time()
        losses, total_correct = train_epoch(epoch, model, train_loader, optimizer, log_tensorboard)

        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        average_loss = np.mean(np.array(losses))
        accuracy = total_correct/dataset_length

        if log_tensorboard:
            writer.add_scalar("Attack Loss/train", average_loss, epoch)
            writer.add_scalar("Attack Correct", total_correct, epoch)
            writer.add_scalar("Attack Accuracy", accuracy, epoch)

        train_losses.append(average_loss)
        train_accuracies.append(accuracy)

        writer.flush()
        writer.close()

    return model, train_losses, train_accuracies

def test(model, model_params, idx, test_loader):
    # load saved model
    model.load_state_dict(model_params)
    model.eval()
    dataset_length = len(test_loader.dataset)
    model.to(DEVICE)

    test_losses = []
    with torch.no_grad():
        total_correct = 0
        for iter, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.type(torch.FloatTensor).to(DEVICE)
            labels = labels.type(torch.FloatTensor).to(DEVICE)

            outputs = model(inputs)

            outputs = outputs.type(torch.FloatTensor).to(DEVICE)
            labels = labels.type(torch.LongTensor).to(DEVICE)

            loss = criterion(outputs, labels)
            test_losses.append(loss.detach().item())
            total_correct += get_num_correct(outputs, labels)

        test_accuracy = total_correct/dataset_length
        test_loss = np.mean(np.array(test_losses))

    return test_loss, test_accuracy

def get_attributions(model, model_params, idx, loader):
    model.load_state_dict(model_params)
    model.eval()
    model.to(DEVICE)

    attributions = []

    with torch.no_grad():
        for iter, (inputs, _) in enumerate(loader):
            inputs = inputs.type(torch.FloatTensor).to(DEVICE)

            outputs = model(inputs)

            outputs = outputs.type(torch.FloatTensor).to(DEVICE)

            outputs_softmax = nn.functional.softmax(outputs, dim = 1)
            preds = torch.argmax(outputs_softmax, dim = 1)

            inputs_preds = list(zip(inputs, preds))

            # is the attribution for the true class or predicted class?
            # The paper claims it does an explanation for a point of interest (y bar)

            attr = list(map(lambda input_pred:
                interpretability.smooth_grad(model, input_pred[0], input_pred[1].item()),
                inputs_preds))

            attributions += attr

        return attributions
