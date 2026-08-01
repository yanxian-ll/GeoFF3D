import open3d as o3d
import numpy as np
import torch
from scipy.linalg import null_space

def to_homogeneous(X):
    return np.hstack([X, np.ones((X.shape[0], 1))])

def apply_homography(H, X, debug=False):
    X_h = to_homogeneous(X)
    X_trans = (H @ X_h.T).T
    if debug:
        print(X_trans[:, 3])
    return X_trans[:, :3] / X_trans[:, 3:]

def apply_homography_batch(H_batch: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """
    Efficiently apply batched 4x4 homographies to 3D points.
    
    Args:
        H_batch: Tensor of shape (B, 4, 4)
        X:       Tensor of shape (N, 3)
    Returns:
        Transformed points: Tensor of shape (B, N, 3)
    """
    B = H_batch.shape[0]
    N = X.shape[0]
    
    # Append 1 to each point: (N, 4)
    ones = torch.ones((N, 1), dtype=X.dtype, device=X.device)
    X_h = torch.cat([X, ones], dim=1)  # (N, 4)

    # Apply homographies: (B, 4, 4) x (N, 4)^T → (B, 4, N)
    X_h = X_h.T.unsqueeze(0).expand(B, 4, N)  # (B, 4, N)
    X_trans = torch.bmm(H_batch, X_h)  # (B, 4, N)

    # Perspective divide
    X_trans = X_trans[:, :3, :] / X_trans[:, 3:4, :]  # (B, 3, N)
    
    # Transpose to (B, N, 3)
    return X_trans.permute(0, 2, 1)

def estimate_3D_homography(X_src_batch, X_dst_batch):
    """
    Estimate batch of 3D Homography.
    
    Inputs:
        X_src_batch: (B, N, 3)
        X_dst_batch: (B, N, 3)
        
    Returns:
        H_batch: (B, 4, 4)
    """
    B, N, _ = X_src_batch.shape
    ones = np.ones((B, N))

    x, y, z = X_src_batch[:, :, 0], X_src_batch[:, :, 1], X_src_batch[:, :, 2]
    xp, yp, zp = X_dst_batch[:, :, 0], X_dst_batch[:, :, 1], X_dst_batch[:, :, 2]

    # Prepare matrices
    A = np.zeros((B, 3 * N, 16))

    stacked_X = np.stack([x, y, z, ones], axis=2)  # (B, N, 4)

    # Fill in A
    A[:, 0::3, 0:4] = -stacked_X
    A[:, 0::3, 12:16] = np.stack([x * xp, y * xp, z * xp, xp], axis=2)

    A[:, 1::3, 4:8] = -stacked_X
    A[:, 1::3, 12:16] = np.stack([x * yp, y * yp, z * yp, yp], axis=2)

    A[:, 2::3, 8:12] = -stacked_X
    A[:, 2::3, 12:16] = np.stack([x * zp, y * zp, z * zp, zp], axis=2)

    # Solve using null space
    H_batch = np.zeros((B, 4, 4))
    for i in range(B):
        nullvec = null_space(A[i])
        if nullvec.shape[1] == 0:
            H_batch[i] = np.eye(4)
            continue

        H = nullvec[:, 0].reshape(4, 4)
        if H[3, 3] == 0:
            H_batch[i] = np.eye(4)
            continue

        H = H / H[3, 3]

        det = np.linalg.det(H)
        if np.isnan(det) or det < 0.0001:
            H_batch[i] = np.eye(4)
        else:
            H_batch[i] = H / det**0.25

    return torch.tensor(H_batch, dtype = torch.float32, device='cuda')

def is_planar(X, threshold=5e-2):
    X_centered = X - X.mean(axis=0)
    _, S, _ = np.linalg.svd(X_centered)
    normal_strength = S[-1] / S[0]
    return normal_strength < threshold

def scale(X):
    centroid = X.mean(axis=0)
    X_centered = X - centroid  # move centroid to origin

    # Compute average distance to the origin after centering
    avg_norm = np.linalg.norm(X_centered, axis=1).mean()

    # Desired average distance is sqrt(3)
    desired_avg_norm = np.sqrt(3)

    # Compute the uniform scaling factor
    scale = desired_avg_norm / avg_norm

    # Construct the 4x4 similarity transform matrix
    T = np.eye(4)
    T[:3, :3] *= scale  # apply scaling
    T[:3, 3] = -scale * centroid  # apply translation

    X_h = np.hstack([X, np.ones((X.shape[0], 1))])  # shape: (N, 4)

    # Step 2: Apply the transform
    X_transformed_h = (T @ X_h.T).T  # shape: (N, 4)

    # Step 3: Convert back to 3D (drop the homogeneous coordinate)
    X_transformed = X_transformed_h[:, :3]

    return T, X_transformed

def ransac_projective(X1_np, X2_np, threshold=0.01, max_iter=300, sample_size=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert to torch tensors on GPU
    X1 = torch.tensor(X1_np, dtype=torch.float32, device=device)
    X2 = torch.tensor(X2_np, dtype=torch.float32, device=device)
    N = X1.shape[0]

    # Sample indices for each hypothesis.
    indices = torch.randint(0, N, (max_iter, sample_size), device=device)

    # Gather sampled point sets.
    X1_samples = torch.stack([X1[idx] for idx in indices])  # (max_iter, sample_size, 3)
    X2_samples = torch.stack([X2[idx] for idx in indices])  # (max_iter, sample_size, 3)

    # Estimate homographies.
    H_ests = estimate_3D_homography(X1_samples.cpu().numpy(), X2_samples.cpu().numpy())

    # Apply homographies to all points.
    X2_preds = apply_homography_batch(H_ests, X1)

    # Compute Euclidean error.
    errors = torch.norm(X2_preds - X2[None, :, :], dim=2)

    # Compute inlier masks and counts.
    inlier_masks = errors < threshold  # (max_iter, N)
    inlier_counts = inlier_masks.sum(dim=1)

    # Select best hypothesis
    best_idx = torch.argmax(inlier_counts)
    best_H = H_ests[best_idx].cpu().numpy()

    return best_H