import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def subsample(X, y, subsample_size, num_sets, is_disjoint=True):
    total_sample = subsample_size * num_sets
    indices = np.random.choice(
            X.shape[0],
            total_sample,
            replace = not is_disjoint
            )
    np.random.shuffle(indices)

    random_X = X[indices]
    random_y = y[indices]

    X_sets = np.split(random_X, num_sets)
    y_sets = np.split(random_y, num_sets)

    return list(zip(X_sets, y_sets))

def plot_losses(losses):
    epochs = np.arange(len(losses[0]))
    for idx, model_losses in enumerate(losses):
        plt.plot(epochs, model_losses, label=f'Model {str(idx + 1)}')
    plt.xlabel('Number of Epochs', fontsize=14)
    plt.ylabel('Cross Entropy Loss', fontsize=14)
    plt.title('Plot of Loss vs Epochs', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    fig = plt.gcf()
    fig.set_size_inches(8.25, 4.5)
    plt.legend(loc="upper right")
    plt.show()

def plot_accuracies(accuracies):
    epochs = np.arange(len(accuracies[0]))
    for idx, model_acc in enumerate(accuracies):
        plt.plot(epochs, model_acc, label=f'Model {str(idx + 1)}')
    plt.xlabel('Number of Epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Plot of Accuracy vs Epochs', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    fig = plt.gcf()
    fig.set_size_inches(8.25, 4.5)
    plt.legend(loc="lower right")
    plt.show()
