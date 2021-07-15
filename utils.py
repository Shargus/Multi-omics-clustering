import os
import gdown
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy as sp
import matplotlib.cm as cm
from itertools import product

from collections import Counter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import silhouette_score, confusion_matrix, adjusted_mutual_info_score, adjusted_rand_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance_matrix

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.layers import Dense, Input, BatchNormalization, LeakyReLU
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import MeanSquaredError as MAE
from tensorflow.keras.regularizers import L1, L2




##### PLOT FUNCTIONS

def plot_2D_dataset(X, cluster_labels, cluster_centers=None, title="2D dataset", caption=None):
    """
    Plot a 2D dataset with cluster labels (and eventually cluster centers)
    Input
        X: N*M 2D dataset
        cluster_labels: the N cluster assignments
        cluster_centers: the n_cluster cluster centers
        title: the title of the visualization
        caption: the subtitle of the visualization
    """

    fig, ax = plt.subplots()

    n_clusters = len(np.unique(cluster_labels))

    ##### Plot the 2D dataset (same cluster <=> same color)
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    if np.any(np.unique(cluster_labels) < 0):
        # case in which a cluster label -1 is present
        colors = cm.nipy_spectral( (cluster_labels + 1).astype(float) / n_clusters)
    ax.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')

    ##### Plot the cluster centers, if given
    if cluster_centers is not None:
        # Draw white circles at cluster centers
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='o', c="white", alpha=1, s=200, edgecolor='k')
        # Draw numbers at cluster centers (i.e. label the clusters)
        for i, c in enumerate(cluster_centers):
            ax.scatter(c[0], c[1], marker='$%d$' %i, alpha=1, s=50, edgecolor='k')

    ax.set_title(caption)
    ax.set_xlabel("Feature space for the 1st feature")
    ax.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.show()


def plot_confusion_matrix(true_cluster_labels, pred_cluster_labels, title="Confusion matrix of the cluster labels (true and predicted)"):
    """
    Plot the confusion matrix as a heatmap, given true cluster labels and predicted labels
    Input: 
        true_cluster_labels: true labels of the data points
        pred_cluster_labels: predicted labels of the data points, from the clustering algorithm
    """

    fig = plt.figure(figsize=(30,10))
    ax = fig.add_subplot(111)

    # Definition of the confusion matrix
    conf_mat = confusion_matrix(true_cluster_labels, pred_cluster_labels)

    # Plot the heatmap representing the confusion matrix
    ax = sns.heatmap(conf_mat,square=True, annot=True, fmt="d", annot_kws={"size":80 / np.sqrt(len(conf_mat))}, cbar=False)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=20);
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=24)
    plt.xlabel('Predicted cluster label', fontsize=24)
    plt.ylabel('True cluster label', fontsize=20)

    # Compute adjusted Rand index and adjusted mutual info score
    ari = adjusted_rand_score(true_cluster_labels, pred_cluster_labels) # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html
    ami = adjusted_mutual_info_score(true_cluster_labels, pred_cluster_labels)  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
    ax.set_title(f"ARI = {ari:.3f}, AMI = {ami:.3f}", fontsize=28)
    plt.suptitle(title, ha='center', fontsize=28, fontweight='bold')
    plt.show()





##### TRAINING UTILITY FUNCTIONS

def cartesian(*args):
    """
    Returns the cartesian product (as a list of tuples) of the given inputs
    """
    return list(product(*[l for l in args]))


def find_closest_cluster_centers(encoded_batch, cluster_cen):
    """
    Find closest cluster center for each sample of the batch in the latent space,
    given encoded batch and cluster centers.
    Input
        encoded_batch: the latent space associated with the given batch
        cluster_cen: cluster centers related to the given batch on the latent space
    """
    closest_cluster_centers = []
    for point in encoded_batch:
        # Compute the euclidean distance between the current encoded point and each centroid
        dist = (point - cluster_cen)**2
        dist = np.sum(dist, axis=1)
        dist = np.sqrt(dist)
        idx = np.argmin(dist)
        closest_cluster_centers.append(cluster_cen[idx])
    return np.array(closest_cluster_centers)


def custom_loss(input_batch, decoded_batch, encoded_batch=None, cluster_cen=None, lmbd=0.1, phase=1):
    """
    Loss of the model
    Input
        input_batch: input batch on which compute the loss
        decoded_batch: output of the model given the input batch
        encoded_batch: output of the encoder given the input batch (needed for phase 2)
        cluster_cen: cluster centers on the latent space (needed for phase 2)
        lmbd: lambda parameter, weight of the closest cluster center loss (needed for phase 2)
        phase: training phase (1st or 2nd)
    """

    mae = MAE()
    # Phase 1: only reconstruction loss
    if phase == 1:
            loss = mae(input_batch, decoded_batch)
            return loss
    # Phase 2: reconstruction loss + closest cluster center loss
    else:
        closest = find_closest_cluster_centers(encoded_batch, cluster_cen)       # (BATCH_SIZE, LS_DIM)
        input_batch = tf.cast(input_batch, tf.float64)
        decoded_batch = tf.cast(decoded_batch, tf.float64)
        encoded_batch = tf.cast(encoded_batch, tf.float64)

        reconstruction_loss = mae(input_batch, decoded_batch)
        closest_cluster_loss = lmbd*mae(encoded_batch,closest)
        loss = reconstruction_loss + closest_cluster_loss

        return loss, tf.cast(reconstruction_loss, tf.float32), tf.cast(closest_cluster_loss, tf.float32)


def Callback_EarlyStopping(LossList, min_delta=0.01, patience=20):
    """
    Check if early stopping conditions are met, returning either True or False
    Input:
        LossList: the list containing the losses of each epoch performed so far
        min_delta: the relative change threshold
        patience: the last patience epochs are compared with the second to last patience epochs
    """
    #No early stopping for 2*patience epochs 
    if len(LossList)//patience < 2 :
        return False
    #Mean loss for last patience epochs and second-last patience epochs
    mean_previous = np.mean(LossList[::-1][patience:2*patience]) #second-last
    mean_recent = np.mean(LossList[::-1][:patience]) #last
    #you can use relative or absolute change
    delta_abs = np.abs(mean_recent - mean_previous) #abs change
    delta_abs = np.abs(delta_abs / mean_previous)  # relative change
    if delta_abs < min_delta :  # if the relative change is less than min_delta then early-stop
        print("Loss didn't change much from last %d epochs"%(patience))
        print("Percent change in loss value:", delta_abs*1e2)
        return True
    else:
        return False