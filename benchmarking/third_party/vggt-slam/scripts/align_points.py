import open3d as o3d
import numpy as np

def preprocess_point_cloud(pcd, voxel_size):
    """Downsample and compute FPFH features."""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )
    return pcd_down, fpfh

def visualize_alignment(source, target, transformation):
    """Visualize the aligned point clouds."""
    source_transformed = source.transform(transformation)

    source.paint_uniform_color([1, 0, 0])  # Red
    target.paint_uniform_color([0, 1, 0])  # Green

    o3d.visualization.draw_geometries([source_transformed, target], window_name="Aligned Point Clouds")

def register_point_clouds(source_np, target_np, voxel_size=0.05):
    """Find similarity transform (rotation, translation, and scale) between two point clouds."""

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_np)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_np)

    # Preprocess
    src_down, src_fpfh = preprocess_point_cloud(source, voxel_size)
    tgt_down, tgt_fpfh = preprocess_point_cloud(target, voxel_size)

    # Global registration using RANSAC
    distance_threshold = voxel_size * 1.5
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
        4,  # RANSAC iterations
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )

    # Refine with ICP
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
    )

    # Extract transformation components
    transformation = result_icp.transformation
    R = transformation[:3, :3].copy()  # Make a writable copy
    t = transformation[:3, 3]          # Translation remains fine
    s = np.cbrt(np.linalg.det(R))      # Compute scale
    R /= s  # Normalize rotation to remove scale
    transformation = result_icp.transformation
    # visualize_alignment(source, target, transformation)

    return R, t, s, transformation