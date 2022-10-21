import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
###plt.style.use("seaborn")
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

def load_embeddings():
    """
    Loading the wine dataset in pandas dataframe
    :return: scaled data
    """
    # loading wine dataset

    wine_raw = pd.read_csv("wine-clustering.csv")

    # checking data shape
    row, col = wine_raw.shape
    print(f'There are {row} rows and {col} columns') 
    print(wine_raw.head(10))

    # to work on copy of the data
    wine_raw_scaled = wine_raw.copy()
    print(wine_raw_scaled.describe())
    wine_raw_scaled.describe().to_csv()
    # Scaling the data to keep the different attributes in same range.
    wine_raw_scaled[wine_raw_scaled.columns] = StandardScaler().fit_transform(wine_raw_scaled)
    print(wine_raw_scaled.describe())

    return wine_raw_scaled


def pca_embeddings(df_scaled):
    """To reduce the dimensions of the wine dataset we use Principal Component Analysis (PCA).
    Here we reduce it from 13 dimensions to 3.

    :param df_scaled: scaled data
    :return: pca result, pca for plotting graph
    """

    pca_3 = PCA(n_components=3)
    pca_3_result = pca_3.fit_transform(df_scaled)
    print('Explained variation per principal component: {}'.format(pca_3.explained_variance_ratio_))
    print('Cumulative variance explained by 3 principal components: {:.2%}'.format(
        np.sum(pca_3.explained_variance_ratio_)))
    return pca_3_result, pca_3


def kmean_hyper_param_tuning(data, range_n_clusters):
    """
    Hyper parameter tuning to select the best from all the parameters on the basis of silhouette_score.

    :param data: dimensionality reduced data after applying PCA
    :return: best number of clusters for the model (used for KMeans n_clusters)
    """
    # instantiating ParameterGrid, pass number of clusters as input
    parameter_grid = ParameterGrid({'n_clusters': range_n_clusters})

    best_score = -1
    kmeans_model = KMeans()     # instantiating KMeans model
    silhouette_scores = []

    # evaluation based on silhouette_score
    for p in parameter_grid:
        kmeans_model.set_params(**p)    # set current hyper parameter
        kmeans_model.fit(data)          # fit model on wine dataset, this will find clusters based on parameter p

        ss = metrics.silhouette_score(data, kmeans_model.labels_)   # calculate silhouette_score
        silhouette_scores += [ss]       # store all the scores

        #print('Parameter:', p, 'Score', ss)

        # check p which has the best score
        if ss > best_score:
            best_score = ss
            best_grid = p

    # plotting silhouette score
    # plt.bar(range(len(silhouette_scores)), list(silhouette_scores), align='center', color='blue', width=0.5)
    # plt.plot(range(len(silhouette_scores)), list(silhouette_scores), '-bo')
    # plt.xticks(range(len(silhouette_scores)), list(range_n_clusters))
    # plt.title('Silhouette Score', fontweight='bold')
    # plt.xlabel('Number of Clusters')
    # plt.show()

    return best_grid['n_clusters']


def visualizing_results(pca_result, label, centroids_pca):
    """ Visualizing the clusters

    :param pca_result: PCA applied data
    :param label: K Means labels
    :param centroids_pca: PCA format K Means centroids
    """
    # ------------------ Using Matplotlib for plotting-----------------------
    x = pca_result[:, 0]
    y = pca_result[:, 1]
    z = pca_result[:, 2]

    ### mpl.rcdefaults()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(x, y, z, c=label, alpha=0.5, s=50)  # plot different colors per cluster
    # plt.scatter(x, y, c=label, alpha=0.5, s=100)  # plot different colors per cluster

    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Clusters")
    ax.add_artist(legend1)

    plt.title('Wine clusters')
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=50, linewidths=1.5, color='red', lw=1.5)

    fig3 = plt.figure()
    ax2 = fig3.add_subplot(111, projection='3d', proj_type='ortho')
    scatter1 = ax2.scatter(x, y, z, c=label, alpha=0.5, s=50)  # plot different colors per cluster
    legend2 = ax2.legend(*scatter1.legend_elements(), loc="upper right", title="Clusters")
    ax2.add_artist(legend2)

    plt.title('Wine clusters')
    ax2.set_xlabel('PCA 1')
    ax2.set_ylabel('PCA 2')
    ax2.set_zticks([])
    ax2.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=50, linewidths=1.5, color='red', lw=1.5)
    ax2.view_init(90, 90)



    plt.show()

# ==================================================================================

def sil_visual(data, range_n_clusters):
    """
    This function is to plot silhouette_score
    :param data:
    :param range_n_clusters:
    """
    # ax = ax.flatten()
    # fig.set_size_inches(18, 7)
    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]

    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    sil_score_avg = []
    fig1, ax1 = plt.subplots()
    sil_score_avg = []
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, ax = plt.subplots(1, 1)
        #fig.set_size_inches(18, 7)
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
    x = [i for i in range(2, len(sil_score_avg) + 2)]
    ax1.plot(x, sil_score_avg, '-bo')
    ax1.set_title('Silhouette Score', fontweight='bold')
    ax1.set_xlabel('Number of Clusters')
    ax1.grid(True)
    plt.show()


def main():
    range_n_clusters = np.arange(2, 11, 1).tolist()
    print("1. Loading Wine dataset")
    data_scaled = load_embeddings()

    print("2. Reducing via PCA")
    pca_result, pca_3 = pca_embeddings(data_scaled)

    print("3. HyperTuning the Parameter for KMeans")
    optimum_num_clusters = kmean_hyper_param_tuning(pca_result, range_n_clusters)
    print("optimum num of clusters =", optimum_num_clusters)
    # fitting KMeans
    kmeans = KMeans(n_clusters=optimum_num_clusters, random_state=10)
    kmeans.fit(data_scaled)
    centroids = kmeans.cluster_centers_
    centroids_pca = pca_3.transform(centroids)

    # Visualize silhouette score
    sil_visual(pca_result, range_n_clusters)
    print("4. Visualizing the data")
    visualizing_results(pca_result, kmeans.labels_, centroids_pca)


if __name__ == "__main__":
    main()

