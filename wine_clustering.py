import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

def load_embeddings():
    """
    Loading the wine dataset in pandas dataframe
    :return: scaled data
    """
    # loading wine dataset

    wine_raw = pd.read_csv("wine-clustering.csv")
    # to work on copy of the data
    wine_raw_scaled = wine_raw.copy()

    # Scaling the data to keep the different attributes in same range.
    wine_raw_scaled[wine_raw_scaled.columns] = StandardScaler().fit_transform(wine_raw_scaled)
    return wine_raw_scaled


def sil_visual(data, range_n_clusters):
    sil_score_avg = []
    fig1, ax1 = plt.subplots()
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 1 columns
        fig, ax = plt.subplots(1, 1)
        #fig.set_size_inches(10, 7)
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(data)
        ax.set_xlim([-0.2, 1])
        ax.set_ylim([0, len(data) + (n_clusters + 1) * 10])
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(data, cluster_labels)
        sil_score_avg.append(silhouette_avg)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(data, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax.set_title("The silhouette plot for the various clusters.")
        ax.set_xlabel("The silhouette coefficient values")
        ax.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax.set_yticks([])  # Clear the yaxis labels / ticks
        ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

    x = [i for i in range(2, len(sil_score_avg)+2)]
    ax1.plot(x, sil_score_avg, '-bo')
    ax1.set_title('Silhouette Score', fontweight='bold')
    ax1.set_xlabel('Number of Clusters')
    ax1.grid(True)
    plt.show()


def main():
    range_n_clusters = np.arange(2, 11, 1).tolist()
    print("1. Loading Wine dataset")
    data_scaled = load_embeddings()

    # Visualize silhouette score
    sil_visual(data_scaled, range_n_clusters)


if __name__ == "__main__":
    main()



