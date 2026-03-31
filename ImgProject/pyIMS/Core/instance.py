import numpy as np
from numpy.typing import NDArray


class Instantiator():
    def __init__(self, ):
        pass
    
    def pole(points: NDArray[np.float32], labels: NDArray[np.uint8]):

        # for pole
        pole_mask = (labels == 9)
        pole_points = points[pole_mask]# Set DBSCAN parameters (eps is the radius, min_samples is the minimum points to form a cluster)
        eps_distance = 0.3   # example epsilon in the same units as the point coordinates (e.g., meters)
        min_points = 100      # example minimum cluster size (adjust based on density/noise)
        
        # Perform DBSCAN clustering
        print('Pole object clustering ...')
        import cupy as cp
        from cuml.cluster import DBSCAN
        db = DBSCAN(eps=eps_distance, min_samples=min_points)
        pole_points = cp.asarray(pole_points)
        instance_id = db.fit_predict(pole_points)
        instance_id = instance_id.get()

        return instance_id

    def cylinder(points: NDArray[np.float32]):
        _, _, vh = np.linalg.svd(points - points.mean(0), full_matrices= False)
        axis_dir = vh[0]

        proj_len = points @ axis_dir
        points = points - np.outer(proj_len, axis_dir)
        xy = points[:, :2]
        
        from skimage.measure import CircleModel, ransac

        model_robust, inliers = ransac(xy, )

    def __call__(self, points: NDArray[np.float32], labels: NDArray[np.uint8]) -> NDArray[np.int32]:
        
        # TODO objectify return object_id (num_points)
        self.pole(points, labels)

        return instance_id.get()