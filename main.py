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
        distances = torch.ones(points.shape[0], 1) * float('inf')
        distances = distances
        for i in range(self.centers.shape[0]):
            difference = torch.norm(self.centers[i, :] - points[:, :3], dim=1)
            tmp_distance = difference - self.radii[i, :]
            tmp_distance = tmp_distance.reshape(points.shape[0], 1)
            distances = torch.min(distances, tmp_distance)
        return distances


def march(origins: torch.Tensor, directions: torch.Tensor, transform: torch.Tensor, objects: List[ObjectGroup],
          epsilon=1e-2, maximum_iterations=100, debug=False, maximum_distance=1000.0):
    transformed_rays = origins @ transform
    rows = transformed_rays.shape[0]
    distances = torch.ones(rows, 1) * float('inf')
    distances = distances
    mask = torch.zeros(rows, 1).bool()
    acc = torch.zeros_like(distances)
    if debug:
        points = transformed_rays.tolist()
    for _ in range(maximum_iterations):
        for group in objects:
            current_distances = group.nearest(transformed_rays).reshape((rows, 1))
            distances = torch.min(distances, current_distances)
            mask[distances < epsilon] = True
            distances[mask] = 0
            acc += distances
            mask[acc > maximum_distance] = True
            transformed_rays[:, :3] += directions * distances
            if debug:
                points += transformed_rays.tolist()
        if mask.all():
            break
        print(_)
    if debug:
        points_ = np.array(points)
        vis = open3d.geometry.PointCloud(
            points=open3d.utility.Vector3dVector(points_[:, :3])
        )
        open3d.visualization.draw_geometries([vis])
    infinity = torch.ones(rows, 1)
    infinity[~mask] = float('inf')
    infinity[acc > maximum_distance] = float('inf')
    transformed_rays += infinity
    return transformed_rays


def main():
    WIDTH = 400
    HEIGHT = 400
    sphere_group = SphereGroup()
    sphere_group.add_sphere([0, 0, 4], 20)
    sphere_group.add_sphere([45, 0, 800], 19)
    torch.cuda.set_device(0)
    rays = torch.Tensor([
        [x, y, 0, 1]
        for x in range(-WIDTH//2, WIDTH//2)
        for y in range(-WIDTH//2, HEIGHT//2)
    ])

    directions = torch.zeros_like(rays)
    directions = directions[:, :3]
    directions[:, 2] = 1

    res = march(rays, directions, torch.eye(4, 4), [sphere_group], debug=False, maximum_iterations=1000,
                maximum_distance=100)
    res[res == float('inf')] = 0.0
    res = res.reshape(WIDTH, HEIGHT, -1)
    res = torch.norm(res, dim=2) / torch.max(res)
    res *= 255
    import matplotlib.pyplot as plot
    plot.imshow(res)
    plot.show()


if __name__ == '__main__':
    main()
