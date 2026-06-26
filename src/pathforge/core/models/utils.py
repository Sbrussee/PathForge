
import torch

def perform_kmeans(
    x: torch.Tensor,
    num_clusters: int,
    n_iter: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simple differentiable K-Means for RRT-MIL.
    x: (N, D)
    Returns: assignments (N,), centers (K, D)
    """
    N, D = x.shape
    
    # Init centers randomly
    indices = torch.randperm(N)[:num_clusters]
    centers = x[indices] # (K, D)
    
    for i in range(n_iter):
        # (N, K) distance
        dists = torch.cdist(x, centers)
        
        # Assign
        assignments = torch.argmin(dists, dim=1) # (N,)
        
        # Update
        new_centers = []
        for k in range(num_clusters):
            mask = (assignments == k)
            if mask.sum() > 0:
                new_centers.append(x[mask].mean(dim=0))
            else:
                # Re-init empty cluster
                new_centers.append(x[torch.randint(0, N, (1,))].squeeze(0))
        centers = torch.stack(new_centers)
        
    return assignments, centers
