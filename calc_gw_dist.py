import ot

def calculate_gw_distance(node_features_1, distance_1, node_features_2, distance_2, alpha=0.5):
    
    # Define Distributions
    # We assume uniform distribution (each point has equal weight)
    p = ot.unif(node_features_1.shape[0])
    q = ot.unif(node_features_2.shape[0])

    # Calculate the Feature Distance Matrix (M)
    # M[i, j] is the distance between feature vector of node i in G1 and node j in G2
    # metric='euclidean' is standard, but you can use 'sqeuclidean', 'cosine', etc.
    M = ot.dist(node_features_1, node_features_2, metric='euclidean')

    # The alpha parameter controls the trade-off:
    # alpha = 0: Pure Gromov-Wasserstein (Structure only)
    # alpha = 1: Pure Wasserstein (Features only)
    # 0 < alpha < 1: Fused (Both)
    
    # fused_gromov_wasserstein2 returns the actual distance value (scalar)
    gw_dist = ot.gromov.fused_gromov_wasserstein2(
        M,
        distance_1, 
        distance_2, 
        p, 
        q, 
        loss_fun='square_loss',
        alpha=alpha,
        verbose=False,
    )
    
    return gw_dist