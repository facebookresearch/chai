import os
import json
import argparse
import numpy as np
from collections import defaultdict
from collections import Counter
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def parse_args(parser):
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--num_clusters", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=32)
    args = parser.parse_args()
    return args

# Takes input as attention scores, applies offline cluster and
# generates pdf plots for elbow plot analysis per layer
# to determine number of clusters
def main_kmeans_avg(args):
    """
    K-Means
    """
    # Take 500 samples concatenate them and create clusters of shape (32, x)
    # perform k-means clustering and get an error plot
    for layer_num in range(args.num_layers):
        print("Working on Layer {}".format(layer_num))
        file_name = os.path.join(args.path, f"layer_{layer_num}_scores.jsonl")
        concatenate_flag = False
        ln_counter = 0
        errors = [0] * 32
        with open(file_name, "r") as fin:
            for line in fin:
                try:
                    scores = json.loads(line)
                except:
                    print("Line")
                    print(line)
                    import ipdb

                    ipdb.set_trace()
                ln_counter += 1
                scores_numpy = np.array(scores)
                first_head = scores_numpy[0, :]
                num_nonzero = first_head[first_head != 0].shape[0]
                reshaped_data = scores_numpy[:, :num_nonzero]
                # if not concatenate_flag:
                #     # running first time
                #     concatenate_array = reshaped_data
                #     concatenate_flag = True
                # else:
                #     concatenate_array = np.concatenate(
                #         (concatenate_array, reshaped_data), axis=1
                #     )
                # perform clustering
                if ln_counter == 500:
                    break
                for num_clusters in range(32):
                    kmeans = KMeans(
                        n_clusters=num_clusters + 1, random_state=2, n_init="auto"
                    )
                    kmeans.fit(reshaped_data)
                    print("Num Clusters {}".format(num_clusters))
                    print("Error {}".format(kmeans.inertia_))
                    errors[num_clusters] += kmeans.inertia_
                # import ipdb

                # ipdb.set_trace()
        plt.plot(range(1, 33), errors)
        out_figure = f"./{layer_num}_plot.pdf"
        plt.savefig(out_figure, format="pdf")
        plt.close()

def main(args):
    """
    Parse arguments for trainers
    """
    for layer_num in range(args.num_layers):
        print("Working on Layer {}".format(layer_num))
        file_name = os.path.join(args.path, f"layer_{layer_num}_scores.jsonl")
        group_bins = defaultdict(int)
        head_belong_count = defaultdict(list)
        with open(file_name, "r") as fin:
            for line in fin:
                try:
                    scores = json.loads(line)
                except:
                    print("Line")
                    print(line)
                    import ipdb

                    ipdb.set_trace()
                scores_numpy = np.array(scores)
                # NOTE: Avoiding the following approach. Sometimes non zeros are no consisten
                first_head = scores_numpy[0, :]
                num_nonzero = first_head[first_head != 0].shape[0]
                reshaped_cluster = scores_numpy[:, :num_nonzero]
                # nonzeros_scores_numpy = scores_numpy[scores_numpy != 0]
                # reshape_factor = int(nonzeros_scores_numpy.shape[0] / 32)
                dist_arr = cdist(reshaped_cluster, reshaped_cluster, metric="cosine")
                cluster = AgglomerativeClustering(
                    n_clusters=args.num_clusters,
                    metric="precomputed",
                    linkage="average",
                    compute_distances=True,
                )
                cluster = cluster.fit(dist_arr)
                cluster_assignment = cluster.labels_
                for cluster_idx in range(args.num_clusters):
                    grouped_heads = np.where(cluster_assignment == cluster_idx)[
                        0
                    ].tolist()
                    grouped_heads_str = json.dumps(grouped_heads)
                    group_bins[grouped_heads_str] += 1
                    for headnum in grouped_heads:
                        head_belong_count[headnum].append(grouped_heads_str)

                # extract non zero
        counted_heads = []
        out_cluster = []
        # import ipdb

        # ipdb.set_trace()
        for head_id in range(args.num_heads):
            if head_id in counted_heads:
                continue
            head_membership = head_belong_count[head_id]
            most_common = Counter(head_membership).most_common(1)[0][0]
            most_common_list = json.loads(most_common)
            counted_heads.extend(most_common_list)
            out_cluster.append(most_common_list)
        out_file = os.path.join(args.path, f"layer_{layer_num}_out_cluster.json")
        with open(out_file, "w") as fout:
            json.dump(out_cluster, fout)

        # out_file_name_group_bins = os.path.join(
        #     args.path, f"layer_{layer_num}_group_bins_.jsonl"
        # )


if __name__ == "__main__":
    args = parse_args(
        argparse.ArgumentParser(description="Parse Arguments for static clustering")
    )
    main_kmeans_avg(args)
