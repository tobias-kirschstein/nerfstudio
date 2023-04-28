import torch
from nerfacc import ContractionType, contract, contract_inv
from nerfstudio.utils.math import contract_points


def test_contract():
    aabb = torch.Tensor([[-1, -1, -1], [1, 1, 1]]).float().cuda() * 3
    contraction_type = ContractionType.UN_BOUNDED_SPHERE

    points = torch.randn((100, 3)).cuda()
    points = points * torch.arange(100).unsqueeze(1).cuda()

    contracted_points = contract(points, roi=aabb, type=contraction_type)
    contracted_points_grad = contract_points(points, contraction_type, aabb=aabb, ord=float('inf'))

    undistorted_points = contract_inv(contracted_points, roi=aabb, type=contraction_type)

    print(points)
    print(contracted_points)
    print(contracted_points_grad)
    print(undistorted_points)
    print(contracted_points.max(dim=1))