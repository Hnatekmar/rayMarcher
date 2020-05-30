import numpy as np
import open3d
import torch
from typing import List


class ObjectGroup:
    def nearest(self, points: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class SphereGroup(ObjectGroup):

    def __init__(self):
        self.centers = torch.Tensor()
        self.radii = torch.Tensor()

    def add_sphere(self, center: List[float], radius: float):
        self.centers = torch.cat([self.centers,
                                  torch.Tensor([center])])
        self.radii = torch.cat([self.radii, torch.Tensor([[radius]])])

    def nearest(self, points: torch.Tensor) -> torch.Tensor:
        return torch.norm(self.centers - points[:, :3], dim=1) - self.radii


def march(origins: torch.Tensor, directions: torch.Tensor, transform: torch.Tensor, objects: List[ObjectGroup],
          epsilon=1e-2, maximum_iterations=100, debug=False):
    transformed_rays = origins @ transform
    rows = transformed_rays.shape[0]
    distances = torch.ones(rows, 1) * float('inf')
    mask = torch.zeros(rows, 1).bool()
    points = transformed_rays.tolist()
    for _ in range(maximum_iterations):
        for group in objects:
            current_distances = group.nearest(transformed_rays).reshape((rows, 1))
            distances = torch.min(distances, current_distances)
            mask[distances < epsilon] = True
            distances[mask] = 0
            transformed_rays[:, :3] += directions * distances
            points += transformed_rays.tolist()
            if mask.all():
                break
    if debug:
        points_ = np.array(points)
        vis = open3d.geometry.PointCloud(
            points=open3d.utility.Vector3dVector(points_[:, :3])
        )
        open3d.visualization.draw_geometries([vis])
    infinity = torch.ones(rows, 1)
    infinity[~mask] = float('inf')
    transformed_rays *= infinity
    return transformed_rays


def main():
    sphere_group = SphereGroup()
    sphere_group.add_sphere([0, 0, 0], 5)
    torch.cuda.set_device(0)
    import math
    x = torch.Tensor([
       [math.sin(math.degrees(i)) * 50, math.cos(math.degrees(i)) * 50, 0, 1] for i in range(0, 360, 10)
    ])
    res = march(x, -x[:, :3]/50.0, torch.eye(4, 4), [sphere_group], debug=True)
    res = res[:, :2]
    import matplotlib.pyplot as plt
    plt.plot(res[:, 0], res[:, 1], 'r.')
    plt.plot(x[:, 0], -x[:, 1], 'b.')
    plt.show()


if __name__ == '__main__':
    main()
