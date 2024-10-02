# Membership Inference Attack
import pdb
import random
import copy

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
import pandas as pd
from mu_util import run_epoch


def cm_score(estimator, X, y):
    """
    Compute various metrics using confusion matrix and return accuracy.
    
    Args:
        estimator : object
            Estimator object implementing 'predict'.
        X : array-like or pd.DataFrame, shape (n_samples, n_features)
            Input data.
        y : array-like, shape (n_samples,)
            True labels.
    
    Returns:
        float
            Accuracy score.
    """
    y_pred = estimator.predict(X)
    cnf_matrix = confusion_matrix(y, y_pred)
    
    FP, FN, TP, TN = cnf_matrix[0][1], cnf_matrix[1][0], cnf_matrix[0][0], cnf_matrix[1][1]

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    return ACC


def evaluate_attack_model(sample_loss, members, n_splits=5, random_state=None):
    """
    Computes the cross-validation score of a membership inference attack.

    Args:
        sample_loss : array_like of shape (n,).
            Objective function evaluated on n samples.
        members : array_like of shape (n,).
            Whether a sample was used for training (0 for non-members, 1 for members).
        n_splits: int, optional, default=5
            Number of splits to use in cross-validation.
        random_state: int, RandomState instance or None, optional, default=None
            Random state to use in cross-validation splitting.

    Returns:
        score : array_like of size (n_splits,)
            Cross-validation scores.
    """
    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")

    # Initialize the attack model (Logistic Regression)
    attack_model = LogisticRegression()

    # Stratified cross-validation for evaluating the attack model
    cv = StratifiedShuffleSplit(n_splits=n_splits, random_state=random_state)

    # Compute cross-validation score using cross_val_score
    scores = cross_val_score(attack_model, sample_loss, members, cv=cv, scoring=cm_score)

    return scores

def calculate_losses(model, criterion, data_loader, device):
    model.eval()
    losses = []
    cri_no_reduction = copy.deepcopy(criterion)
    cri_no_reduction.reduction = 'none'
    with torch.inference_mode():
        for image, target in data_loader:
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = cri_no_reduction(output, target)
            losses.extend(loss.cpu().detach().numpy())
    return losses

def calculate_losses2(model, dataloader, loss_fn='ce', multiplier=1.0):
    """
    Calculate losses for a given model and dataloader.

    Args:
        model: torch.nn.Module
            The neural network model.
        dataloader: torch.utils.data.DataLoader
            DataLoader providing the input data.
        loss_fn: str, optional, default='ce'
            Loss function type ('ce' for Cross Entropy, 'mse' for Mean Squared Error).
        multiplier: float, optional, default=1.0
            Multiplier to scale the computed losses.

    Returns:
        list
            List of computed losses.
    """
    losses = []
    criterion = nn.CrossEntropyLoss(reduction='none')
    device = next(model.parameters()).device  # Get the device of the network parameters

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            target = torch.from_numpy(np.asarray(target).astype('long'))
            data, target = data.to(device), target.to(device)
            
            if loss_fn == 'mse':
                target = (2 * target - 1).type(torch.cuda.FloatTensor).unsqueeze(1)
            
            output = model(data)
            # import pdb; pdb.set_trace()
            loss = multiplier * criterion(output, target)
            losses.extend(loss.cpu().detach().numpy())

    return losses


def membership_inference_attack(model, criterion, test_loader, forget_loader, device, seed):
    """
    Conducts a membership inference attack on the given model using test and forget data.

    Args:
        model: torch.nn.Module
            The neural network model.
        test_loader: torch.utils.data.DataLoader
            DataLoader providing test data.
        forget_loader: torch.utils.data.DataLoader
            DataLoader providing forget data.
        loss_fn: str
            Loss function type ('ce' for Cross Entropy, 'mse' for Mean Squared Error).
        seed: int
            Seed for reproducibility.

    Returns:
        float
            Attack score.
    """
    # Filter test loader based on forget_loader classes
    # forget_classes = list(np.unique(forget_loader.dataset.targets))
    # test_indices = [i in forget_classes for i in test_loader.dataset.targets]
    # test_loader.dataset.data = test_loader.dataset.data[test_indices]
    # test_loader.dataset.targets = test_loader.dataset.targets[test_indices]

    # Calculate losses for test and forget data
    model.eval()
    test_losses = calculate_losses(model, criterion, test_loader, device)
    forget_losses = calculate_losses(model, criterion, forget_loader, device)
    # forget_loader.dataset.dataset.transform = train_transform
    min_len = min(len(test_losses), len(forget_losses))

    # repeat the experiment 10 times
    attack_scores = []
    n = 10
    for s in range(seed, seed+n):
        # Ensure equal number of samples for both sets
        np.random.seed(s)
        random.seed(s)
        forget_losses_sample = random.sample(forget_losses, min_len)
        test_losses_sample = random.sample(test_losses, min_len)

        # Prepare data for attack model evaluation
        test_labels = [0] * min_len
        forget_labels = [1] * min_len
        features = np.array(test_losses_sample + forget_losses_sample).reshape(-1, 1)
        labels = np.array(test_labels + forget_labels).reshape(-1)
        features = np.clip(features, -100, 100)

        # Evaluate attack model and return score
        attack_score = evaluate_attack_model(features, labels, n_splits=10, random_state=s)
        attack_scores.append(np.mean(attack_score))
        # pdb.set_trace()
    return attack_scores


def plot_loss_histograms(test_losses, forget_losses, title='loss_histograms.png'):
    sns.distplot(np.array(test_losses), kde=False, norm_hist=False, rug=False, label='test-loss')
    sns.distplot(np.array(forget_losses), kde=False, norm_hist=False, rug=False, label='forget-loss')
    plt.legend(prop={'size': 14})
    plt.tick_params(labelsize=12)
    plt.title("Loss Histograms", size=18)
    plt.xlabel('Loss Values', size=14)
    plt.savefig(title)
    plt.close()


def print_errors(errors):
    # pdb.set_trace()
    tes = {}
    res = {}
    fes = {}
    ves = {}
    rlt = {}
    MIA = {}

    for key in errors[0].keys():
        tes[key] = [errors[i][key]['test_error'] for i in range(len(errors))]
        fes[key] = [errors[i][key]['forget_error'] for i in range(len(errors))]
        MIA[key] = [errors[i][key]['MIA']*100 for i in range(len(errors))]
        
        print ("{}  \t{:.2f}±{:.2f}\t{:.2f}±{:.2f}\t{:.2f}±{:.2f}".format(key, 
                                                                    np.mean(tes[key]), np.std(tes[key]),
                                                                    np.mean(fes[key]), np.std(fes[key]),
                                                                    np.mean(MIA[key]), np.std(MIA[key])))


def calculate_average_std(data):
    """
    Calculates the average and standard deviation of each key for each method in the given list of dictionaries.

    Parameters:
    - data: List of dictionaries containing keys and values.

    Returns:
    - result: A dictionary containing the average and standard deviation for each key and method.
    """

    method_data = {}
    # Group data by method
    for entry in data:
        for method, values in entry.items():
            if method not in method_data:
                method_data[method] = {key: [values[key]] for key in values}
            else:
                for key, value in values.items():
                    method_data[method].setdefault(key, []).append(value)

    # Calculate average and standard deviation for each key and method
    header = "Method  \t"
    info_list = []
    for method, values in method_data.items():
        info = f"{method}  \t"
        for key, val_list in values.items():
            header += f"{key}\t" if key not in header else ""
            info += f"{np.mean(val_list):.2f}±{np.std(val_list):.2f} \t"
        info_list.append(info)
    return header, info_list
