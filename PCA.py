import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import matplotlib.dates as mdates

from utils.logger import setup_logging, get_logger

import logging
logger = get_logger(__name__)
setup_logging(level=logging.INFO)

def load_data(date_range: tuple|None=None):
    # Load the data
    data = []
    for file in Path("data").glob("*.csv"):
        data.append(pd.read_csv(file, index_col=0, parse_dates=True))
    res = pd.concat(data).sort_index()
    if date_range:
        res = res.loc[date_range[0]:date_range[1]]
    return res

def _drop_na(dat):
    res = dat.drop(['1.5 Month', '2 Mo', '4 Mo'], axis='columns')
    res = res.dropna(axis='rows') # TODO check which rows are dropped
    return res

def preprocess_data(data, report_stats=False):
    # Preprocess the data
    move = data.diff()
    move = _drop_na(move)
    scaler = StandardScaler()
    move_hat = scaler.fit_transform(move)
    move_hat = pd.DataFrame(move_hat, columns=move.columns, index=move.index)
    if report_stats:
        cols = ['2 Yr', '10 Yr']
        logger.info("Preprocessing stats: zero mean & unit variance")
        logger.info(pd.concat([move[cols], move_hat[cols]], axis=1).describe())
    return move_hat

def apply_pca(data, n_components=None, plot=False, axes=None, n_top_loadings=10, figsize=(12,5)):
    """
    Fit PCA to `data`, return (pca, scores, loadings_df, explained_variance_ratio_).
    If `plot` is True, plot explained variance (bar + cumulative line) and a heatmap
    of loadings for the top `n_top_loadings` features (by max absolute loading across PCs).

    Args:
        data: pandas DataFrame or 2D array (samples x features)
        n_components: int or float passed to sklearn.PCA
        scale: whether to StandardScale features before PCA
        plot: whether to show plots
        n_top_loadings: number of features to show in the loadings heatmap
        figsize: figure size for the plots

    Returns:
        pca: fitted sklearn.decomposition.PCA instance
        scores: numpy array or pandas DataFrame (if input was DataFrame) of transformed data
        loadings_df: pandas DataFrame (features x PCs) of component loadings
        explained: numpy array of explained_variance_ratio_
    """
    # detect DataFrame-like input
    is_df = hasattr(data, "values") and hasattr(data, "columns")
    if is_df:
        values = data.values
        feature_names = list(data.columns)
        sample_index = data.index
    else:
        values = np.asarray(data)
        n_features = values.shape[1] if values.ndim == 2 else 1
        feature_names = [f"f{i}" for i in range(n_features)]
        sample_index = None

    # fit PCA
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(values)

    # loadings: features x components
    loadings = pca.components_.T  # shape (n_features, n_components)
    pc_names = [f"PC{i+1}" for i in range(loadings.shape[1])]
    loadings_df = pd.DataFrame(loadings, index=feature_names, columns=pc_names)

    explained = pca.explained_variance_ratio_

    # convert scores to DataFrame if input was DataFrame (keep index)
    if is_df:
        scores_df = pd.DataFrame(scores, index=sample_index, columns=pc_names)
    else:
        scores_df = scores

    logger.info("PCA fitted: %d components, explained variance (first 10): %s",
                loadings.shape[1], np.array2string(explained[:10], precision=4, separator=', '))

    if plot:
        # Explained variance plot
        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1, 1.1]})
        ax = axes[0]
        x = np.arange(1, len(explained) + 1)
        bars = ax.bar(x, explained, alpha=0.7, label='Individual')
        ax.set_xlabel("Principal component")
        ax.set_ylabel("Explained Variance Ratio")
        ax.set_title("Explained Variances by Principle Components")
        ax.set_xticks(x)
        ax.set_yticks(np.arange(0, 1.1, 0.1))

        # add horizontal grid lines for readability
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        # ensure space above tallest bar for annotations
        top = explained.max() if len(explained) > 0 else 1.0
        ax.set_ylim(0, 1.05)

        # annotate each bar with percentage value
        for rect in bars:
            h = rect.get_height()
            ax.annotate(f"{h * 100:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

        # cumulative line on same axes with twin y
        cum = np.cumsum(explained)
        ax.plot(x, cum, color='C1', marker='o', label='Cumulative')
        ax.legend(loc='center right')

        # Loadings heatmap for top features
        axl = axes[1]
        # pick top features by max absolute loading across shown PCs
        n_show_features = min(loadings_df.shape[0], n_top_loadings)
        # choose number of PCs to display (all or up to 10 for readability)
        n_show_pcs = min(loadings_df.shape[1], 5)
        sub = loadings_df.iloc[:n_show_features, :n_show_pcs]

        im = axl.imshow(sub.T, cmap='bwr', aspect='auto',
                        vmin=-np.abs(sub.values).max(), vmax=np.abs(sub.values).max())
        axl.set_yticks(np.arange(n_show_pcs))
        axl.set_yticklabels(pc_names[:n_show_pcs])
        axl.set_xticks(np.arange(len(sub.index)))
        axl.set_xticklabels(sub.index, rotation=90)
        axl.set_title(f"Loadings (top {n_show_features} features)")

        cbar = plt.colorbar(im, ax=axl, fraction=0.046, pad=0.04)
        cbar.set_label("Loading value")

    return pca, scores_df, loadings_df, explained

def outlier_PCA(data, n_components=0.95, threshold_std=3.0, method='mad', plot=False, fig=None, axes=None,
                annotate_top=5, date_range=None):
    """
    Detect outliers using PCA reconstruction error.

    Args:
        data: DataFrame or 2D array (samples x features).
        n_components: int or float (variance ratio) passed to sklearn.PCA.
        threshold_std: multiplier for MAD/std to define threshold.
        method: 'mad' (median absolute deviation, robust) or 'std' (mean/std).
        plot: whether to show a scatter plot of reconstruction errors.
        annotate_top: number of largest errors to annotate on the plot (0 disables).

    Returns:
        outliers: boolean mask or pandas Series (True = outlier)
        reconstruction_errors: numpy array or pandas Series of errors
        threshold: numeric threshold used
    """
    # Normalize input/keep index/columns if DataFrame
    is_df = hasattr(data, "values")
    if is_df:
        values = data.values
        idx = data.index
    else:
        values = np.asarray(data)
        idx = None

    # Fit PCA and reconstruct
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(values)
    reconstructed = pca.inverse_transform(projected)

    # Per-sample Mean Squared Error
    reconstruction_errors = ((values - reconstructed) ** 2).mean(axis=1)

    # Threshold calculation
    if method == 'mad':
        median = np.median(reconstruction_errors)
        mad = 1.4826 * np.median(np.abs(reconstruction_errors - median))
        threshold = median + threshold_std * mad
    else:
        mean = np.mean(reconstruction_errors)
        std = np.std(reconstruction_errors)
        threshold = mean + threshold_std * std

    # Outlier mask
    outlier_mask = reconstruction_errors > threshold

    # Logging
    n_outliers = int(np.sum(outlier_mask))
    total = len(reconstruction_errors)
    logger.info(
        "PCA reconstruction error: n=%d, outliers=%d (%.2f%%), threshold=%.6f",
        total, n_outliers, 100.0 * n_outliers / total, float(threshold)
    )

    # Convert to pandas Series if input was DataFrame
    if is_df:
        outliers = pd.Series(outlier_mask, index=idx)
        reconstruction_errors = pd.Series(reconstruction_errors, index=idx)
    else:
        outliers = outlier_mask

    # Plot
    if plot:
        # Create two-panel figure: left = PC1 vs PC2, right = reconstruction errors over time/index
        if axes is None:
            fig, (ax_time, ax_scores) = plt.subplots(1, 2, figsize=(15, 5))
        else:
            ax_time, ax_scores = axes

        # Scores plot (PC1 vs PC2) if at least 2 components available
        if projected.shape[1] >= 2:
            pc1 = projected[:, 0]
            pc2 = projected[:, 1]
            sc = ax_scores.scatter(pc1, pc2, c=(reconstruction_errors.values if is_df else reconstruction_errors),
                                    cmap='viridis', alpha=0.7)
            # highlight outliers
            if np.any(outlier_mask):
                ax_scores.scatter(pc1[outlier_mask], pc2[outlier_mask],
                                  facecolors='none', edgecolors='r', s=80, linewidths=1.5, label='Outlier')
            ax_scores.set_xlabel("PC1")
            ax_scores.set_ylabel("PC2")
            ax_scores.set_title("PCA Scores (PC1 vs PC2) colored by reconstruction error")
            ax_scores.legend(loc='best')
            plt.colorbar(sc, ax=ax_scores, label='Reconstruction error (MSE)', fraction=0.046, pad=0.04)
        else:
            ax_scores.text(0.5, 0.5, "Less than 2 components; scores plot unavailable",
                           ha='center', va='center')
            ax_scores.set_axis_off()

        ax_time.scatter(reconstruction_errors.index, reconstruction_errors, alpha=0.6)
        ax_time.scatter(outliers.index[outliers], reconstruction_errors[outliers], color='r', alpha=0.6, label='Outliers')
        ax_time.axhline(y=threshold, color='r', linestyle='--', label=f"threshold (MAD, 3 std dev)")
        ax_time.set_xlabel("Sample dates")
        ax_time.set_ylabel("Reconstruction error (MSE)")
        ax_time.set_title("PCA Reconstruction Errors")
        ax_time.set_xlim(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))
        ax_time.grid(axis='y', linestyle='--', alpha=0.5)
        ax_time.legend(loc='upper left')
        ax_time.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax_time.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax_time.get_xticklabels(), rotation=30, ha='right')  # rotate/align labels

        if annotate_top and annotate_top > 0:
            # annotate largest errors
            top_idx = np.argsort(reconstruction_errors.values if is_df else reconstruction_errors)[-annotate_top:]
            for i in top_idx:
                ax_time.annotate(str(reconstruction_errors.index[i].date()),
                             (reconstruction_errors.index[i], reconstruction_errors[i]),
                             textcoords="offset points", xytext=(5,5), fontsize=8)
        plt.tight_layout()
        plt.show()

    return projected, outliers, reconstruction_errors, threshold

def main():
    data = load_data(date_range=('2007-01-01', '2009-01-01'))
    # data = load_data()
    move_std = preprocess_data(data, report_stats=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12)) #, gridspec_kw={"width_ratios": [1, 1.1]})
    pca, scores_df, loadings_df, explained = apply_pca(move_std, plot=True, axes=axes[0])
    outliers, reconstruction_errors, threshold = outlier_PCA(move_std, plot=True, fig=fig, axes=axes[1],
                                                             date_range=('2008-01-01', '2009-01-01'))

if __name__ == "__main__":
    main()