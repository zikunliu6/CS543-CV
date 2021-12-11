# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 13:18:33 2021

@author: kivd
"""

import glob
from scipy.io import wavfile
from scipy.signal import stft
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def PCA(Z, d):
    
    from scipy.linalg import eigh

    n = Z.shape[0]
    
    # Eigan decomposition
    e_vals, e_vecs = eigh(Z, subset_by_index = [n - d, n - 1])

    # Sort the eiganvalues by index
    e_idx_sorted = np.argsort(e_vals)
    e_idx_opp = np.flip(e_idx_sorted)
    
    # Calculate the features
    L = np.diag(1 / (np.sqrt(e_vals) + 1e-6))
    W = L @ e_vecs.T
    
    return np.real(W[e_idx_opp[:d]]), L

def zero_mean(X, axis = 1):
    # Zero mean
    return X - np.mean(X, axis = axis, keepdims = True)
    
def covariance(X, n):
    # Covariance
    # n = digits.shape[1], number of digits
    return 1 / (n - 1) * (X @ X.T)

def features_graph(directory = "../data/", m_Tag=False, m_vale=0):
    
    song_list = glob.glob(directory + "*.npy")
    
    # Look at only half of each song
    # Do PCA
    songs = np.array([np.load(song).flatten() for song in song_list])
    if not m_Tag:
        m = np.mean(songs, axis=0, keepdims=True)
    else:
        m = m_vale
    d = songs - m
    c = covariance(d, d.shape[1])
    wp, L = PCA(c, 2)

    # # Plot PCA-
    # fig = plt.figure()
    # plt.scatter(wp[0], wp[1], s = 1)
    # plt.title("PCA of top 2 components")
    # plt.show()
    #
    # Do TSNE
    X_embedded = TSNE(n_components = 2, init='random').fit_transform(songs).T

    # # Plot TSNE results
    # fig = plt.figure()
    # plt.scatter(X_embedded[0], X_embedded[1], s = 1)
    # plt.title("TSNE of 2 components")
    # plt.show()
    #
    return wp, X_embedded, m

# features_graph("../data/")
for epoch in [0]:
    w_pca_real, w_tsne_real, m = features_graph('../CV_real_data/')
    w_pca_fake, w_tsne_fake, m = features_graph('../CV_generated_data/')

    # Plot PCA-
    fig = plt.figure()
    plt.scatter(w_pca_real[0], w_pca_real[1], s=1)
    plt.scatter(w_pca_fake[0], w_pca_fake[1], s=1)
    plt.title("PCA of top 2 components")
    plt.show()

    # Plot TSNE results
    fig = plt.figure()
    plt.scatter(w_tsne_real[0], w_tsne_real[1], s=1)
    plt.scatter(w_tsne_fake[0], w_tsne_fake[1], s=1)
    plt.title("TSNE of 2 components")
    plt.show()