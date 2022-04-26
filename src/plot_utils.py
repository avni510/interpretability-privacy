import matplotlib.pyplot as plt
import utils as utils
import numpy as np

def plot_losses(losses, plot_dir, is_attack=False, class_val=None):
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
    if is_attack:
        plt.savefig(plot_dir + '/attack_losses_class_' + str(class_val) + '.png')
    else:
        plt.savefig(plot_dir + '/base_losses.png')
    plt.show()

def plot_accuracies(accuracies, plot_dir, is_attack=False, class_val=None):
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
    if is_attack:
        plt.savefig(plot_dir + '/attack_accuracies_class_' + str(class_val) + '.png')
    else:
        plt.savefig(plot_dir + '/base_accuracies.png')
    plt.show()

def plot_train_baseline_models(experiment_dir, iter_val, num_models=4):
    model_infos = utils.load_model_infos(experiment_dir, '/iter_' + str(iter_val), num_models)

    train_losses = [info['training_losses'] for info in model_infos]
    train_accuracies = [info['training_accuracies'] for info in model_infos]

    plt_dir = '/iter_' + str(iter_val) + '/plots'
    utils.create_new_dir(experiment_dir, plt_dir)
    plot_losses(train_losses, experiment_dir + plt_dir)
    plot_accuracies(train_accuracies, experiment_dir + plt_dir)

def display_test_baseline_results(experiment_dir, iter_val, num_models=4):
    print("Test Loss/ Accuracies")
    model_infos = utils.load_model_infos(experiment_dir, '/iter_' + str(iter_val), num_models)

    test_losses = [info['test_loss'] for info in model_infos]
    test_acc = [info['test_accuracy'] for info in model_infos]


    print(f'Losses {test_losses}')
    print(f'Accuracies {test_acc}')

def plot_train_attack_models(experiment_dir, iter_val, num_models=4):
    model_infos = utils.load_model_infos(experiment_dir, '/iter_' + str(iter_val), num_models)

    print("Attack Training Accuracies and Losses")
    classes = [0, 1]
    for class_val in classes:
        print("Plot of class " + str(class_val))

        train_losses = [info['attack_train_losses'][class_val] for info in model_infos]
        train_accuracies = [info['attack_train_accuracies'][class_val] for info in model_infos]

        plt_dir = '/iter_' + str(iter_val) + '/plots'
        utils.create_new_dir(experiment_dir, plt_dir)
        plot_losses(
                train_losses,
                experiment_dir + plt_dir,
                is_attack=True,
                class_val=class_val
                )
        plot_accuracies(
                train_accuracies,
                experiment_dir + plt_dir,
                is_attack=True,
                class_val=class_val
                )

def display_attack_test_results(experiment_dir, iter_val, num_models=4):
    model_infos = utils.load_model_infos(experiment_dir, '/iter_' + str(iter_val), num_models)

    print("Attack Test Accuracies and Losses")
    classes = [0, 1]
    for class_val in classes:
        print("Losses and Accuracies of class " + str(class_val))

        test_losses = [info['attack_test_loss'][class_val] for info in model_infos]
        test_accuracies = [info['attack_test_accuracy'][class_val] for info in model_infos]

        print("Losses")
        print(test_losses)
        print("------------------------------")
        print("Accuracies")
        print(test_accuracies)
        print("")

