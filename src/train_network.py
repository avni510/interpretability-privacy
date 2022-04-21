import numpy as np
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import time
import os
import interpretability as interpretability

EPOCHS = 20
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
# can add the path inside Summary Writer
writer = SummaryWriter()
LR = .001
LR_DECAY= 1e-7

ATTACK_LR = .001
ATTACK_LR_DECAY= 1e-7
ATTACK_EPOCHS = 1

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENT_DIR = ROOT_DIR + '/experiments/adult/'
MODEL_PATH = ROOT_DIR + '/saved_models'

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

        writer.add_graph(model, inputs)

        outputs = model(inputs)

        outputs = outputs.type(torch.FloatTensor).to(DEVICE)
        labels = labels.type(torch.LongTensor).to(DEVICE)

        loss = criterion(outputs, labels)
        losses.append(loss.item())
        total_correct += get_num_correct(outputs, labels)

        if log_tensorboard and iter % 100 == 0:
            writer.add_scalar("Batch Loss", loss.item(), iter)

        #backprop
        loss.backward()

        # update weights
        optimizer.step()

        if iter % 150 == 0:
            print("epoch: {}, iter: {}, loss: {}".format(epoch, iter, loss.item()))
    return losses, total_correct

def train(
        model,
        train_loader,
        idx,
        dataset_length,
        log_tensorboard=False
        ):
    train_losses = []
    train_accuracies = []
    model.to(DEVICE)
    optimizer = torch.optim.Adagrad(
            model.parameters(),
            lr=LR,
            lr_decay=LR_DECAY
           )

    for epoch in range(EPOCHS):
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
        log_tensorboard=False
        ):
    train_losses = []
    train_accuracies = []
    model.to(DEVICE)
    optimizer = torch.optim.Adagrad(
            model.parameters(),
            lr=ATTACK_LR,
            lr_decay=ATTACK_LR_DECAY
           )

    for epoch in range(ATTACK_EPOCHS):
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

def test(model, model_params, idx, test_loader, dataset_length):
    # load saved model
    model.load_state_dict(model_params)
    model.eval()
    test_losses = []
    model.to(DEVICE)

    with torch.no_grad():
        total_correct = 0
        for iter, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.type(torch.FloatTensor).to(DEVICE)
            labels = labels.type(torch.FloatTensor).to(DEVICE)

            outputs = model(inputs)

            outputs = outputs.type(torch.FloatTensor).to(DEVICE)
            labels = labels.type(torch.LongTensor).to(DEVICE)

            loss = criterion(outputs, labels)
            test_losses.append(loss.item())
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
