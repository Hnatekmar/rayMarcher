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
    transformed_rays.cuda()
    rows = transformed_rays.shape[0]
    distances = torch.ones(rows, 1) * float('inf')
    distances.cuda()
    mask = torch.zeros(rows, 1).bool()
    if debug:
        points = transformed_rays.tolist()
    for _ in range(maximum_iterations):
        for group in objects:
            current_distances = group.nearest(transformed_rays).reshape((rows, 1))
            distances = torch.min(distances, current_distances)
            mask[distances < epsilon] = True
            distances[mask] = 0
            transformed_rays[:, :3] += directions * distances
            if debug:
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
    transformed_rays += infinity
    return transformed_rays


def main():
    sphere_group = SphereGroup()
    sphere_group.add_sphere([0, 0, 500], 20)
    torch.cuda.set_device(0)
    rays = torch.Tensor([
        [x, y, 0, 1]
        for x in range(-64, 64)
        for y in range(-64, 64)
    ])

    directions = torch.zeros_like(rays)
    directions = directions[:, :3]
    directions[:, 2] = 1

    res = march(rays, directions, torch.eye(4, 4), [sphere_group], debug=True, maximum_iterations=500)
    res[res == float('inf')] = 0.0
    res = res.reshape(128, 128, -1)
    res = torch.norm(res, dim=2) / torch.max(res)
    res *= 255
    import matplotlib.pyplot as plot
    plot.imshow(res)
    plot.show()


if __name__ == '__main__':
    main()
