import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import wandb


@torch.no_grad()
def visualize_features(
    feat_anchor,
    feat_negative,
    pca_path="pca_plot.png",
    tsne_path="tsne_plot.png",
    saving_dir=None,
):
    """
    Visualize features using PCA and t-SNE.

    Parameters:
    feat_anchor (torch.Tensor): Tensor of anchor features.
    feat_negative (torch.Tensor): Tensor of negative features.
    pca_path (str): File path to save PCA plot.
    tsne_path (str): File path to save t-SNE plot.
    """

    # Convert tensors to numpy arrays
    feat_anchor_np = feat_anchor.detach().cpu().numpy()
    feat_negative_np = feat_negative.detach().cpu().numpy()

    # Combine the features
    features = np.concatenate((feat_anchor_np, feat_negative_np), axis=0)

    # Create labels (1 for anchor, 0 for negative)
    labels = np.concatenate(
        (np.ones(feat_anchor_np.shape[0]), np.zeros(feat_negative_np.shape[0]))
    )

    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)

    # Apply t-SNE
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(features)

    saving_dir = saving_dir if saving_dir else "."
    # Plot PCA
    plt.figure(figsize=(6, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap="viridis", alpha=0.5)

    # plot line between scatters
    for i in range(len(labels) // 2):
        plt.plot(
            [pca_result[i, 0], pca_result[i + len(labels) // 2, 0]],
            [pca_result[i, 1], pca_result[i + len(labels) // 2, 1]],
            c="black",
            alpha=0.1,
        )
    plt.title("PCA Visualization")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar(label="Label")
    pca_path = os.path.join(saving_dir, pca_path)
    plt.savefig(pca_path)
    plt.close()

    # Plot t-SNE
    plt.figure(figsize=(6, 6))
    plt.scatter(
        tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap="viridis", alpha=0.5
    )
    tsne_dist = []
    for i in range(len(labels) // 2):
        plt.plot(
            [tsne_results[i, 0], tsne_results[i + len(labels) // 2, 0]],
            [tsne_results[i, 1], tsne_results[i + len(labels) // 2, 1]],
            c="black",
            alpha=0.1,
        )
        tsne_dist.append(
            np.linalg.norm(tsne_results[i] - tsne_results[i + len(labels) // 2])
        )

    plt.title("t-SNE Visualization")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.colorbar(label="Label")
    tsne_path = os.path.join(saving_dir, tsne_path)
    plt.savefig(tsne_path)
    plt.close()

    return pca_path, tsne_path

    # PL does not support logging plots
    # Object of type CustomChart is not JSON serializable [???]
    pca_data = [[elem[0], elem[1]] for elem in pca_result]
    tsne_data = [[elem[0], elem[1]] for elem in tsne_results]
    pca_pos_table = wandb.Table(
        data=pca_data[: len(pca_data) // 2], columns=["pca_x", "pca_y"]
    )
    pca_neg_table = wandb.Table(
        data=pca_data[len(pca_data) // 2 :], columns=["pca_x", "pca_y"]
    )

    return pca_pos_table, pca_neg_table


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def l2_dist(x1, x2, dim=1, eps=1e-8):
    """Returns l2 distance between x1 and x2, computed along dim."""
    return torch.norm(x1 - x2, 2, dim)


@torch.no_grad()
def stat_feature(
    feat_anchor,
    feat_negative,
    norm_mode="l2",
):
    """_summary_

    Args:
        feat_anchor (_type_): _description_
        feat_negative (_type_): _description_
    """
    feat_anchor = feat_anchor.detach()
    feat_negative = feat_negative.detach()
    # labels = np.concatenate(
    #     (np.ones(feat_anchor_np.shape[0]), np.zeros(feat_negative_np.shape[0]))
    # )

    if norm_mode == "l2":
        compute_dist = l2_dist
    elif norm_mode == "cosine":
        compute_dist = cosine_similarity
    else:
        print("Not implemented")
        raise NotImplementedError
    inter_feat_dist = compute_dist(feat_anchor, feat_negative)
    perm = torch.randperm(feat_anchor.size(0))
    intra_feat_dist_pos = compute_dist(feat_anchor, feat_anchor[perm])
    intra_feat_dist_neg = compute_dist(feat_negative, feat_negative[perm])

    mean_inter = inter_feat_dist.mean().item()
    mean_intra_pos = intra_feat_dist_pos.mean().item()
    mean_intra_neg = intra_feat_dist_neg.mean().item()
    return (
        mean_inter,
        mean_intra_pos,
        mean_intra_neg,
    )
