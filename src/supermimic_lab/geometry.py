"""Differentiable rigid-body operations; every quaternion uses xyzw order."""

import math

import torch


def conjugate(quaternion):
    return torch.cat((-quaternion[..., :3], quaternion[..., 3:]), dim=-1)


def multiply(first, second):
    first_vector, first_scalar = first[..., :3], first[..., 3:]
    second_vector, second_scalar = second[..., :3], second[..., 3:]
    vector = first_scalar * second_vector + second_scalar * first_vector
    vector = vector + torch.cross(first_vector, second_vector, dim=-1)
    scalar = first_scalar * second_scalar - (first_vector * second_vector).sum(
        dim=-1, keepdim=True
    )
    return torch.cat((vector, scalar), dim=-1)


def rotate(quaternion, vector):
    quaternion_vector, vector = torch.broadcast_tensors(quaternion[..., :3], vector)
    first_cross = torch.cross(quaternion_vector, vector, dim=-1)
    return vector + 2 * (
        quaternion[..., 3:] * first_cross
        + torch.cross(quaternion_vector, first_cross, dim=-1)
    )


def rotation_vector_to_quaternion(rotation_vector):
    angle = torch.linalg.vector_norm(rotation_vector, dim=-1, keepdim=True)
    scale = 0.5 * torch.sinc(angle / (2 * math.pi))
    return torch.cat((rotation_vector * scale, torch.cos(angle / 2)), dim=-1)


def quaternion_to_rotation_vector(quaternion):
    quaternion = torch.where(quaternion[..., 3:] < 0, -quaternion, quaternion)
    vector_norm = torch.linalg.vector_norm(quaternion[..., :3], dim=-1, keepdim=True)
    angle = 2 * torch.atan2(vector_norm, quaternion[..., 3:])
    scale = torch.where(
        vector_norm > 1e-6, angle / vector_norm.clamp_min(1e-6),
        2 + vector_norm.square() / 3,
    )
    return quaternion[..., :3] * scale


def orientation_error(predicted, target):
    return quaternion_to_rotation_vector(multiply(conjugate(target), predicted))


def heading(quaternion):
    first, second, third, scalar = quaternion.unbind(dim=-1)
    angle = torch.atan2(
        2 * (scalar * third + first * second),
        1 - 2 * (second.square() + third.square()),
    )
    zeros = torch.zeros_like(angle)
    return torch.stack((zeros, zeros, torch.sin(angle / 2), torch.cos(angle / 2)), dim=-1)


def orientation_columns(quaternion):
    first_axis = torch.zeros_like(quaternion[..., :3])
    second_axis = torch.zeros_like(first_axis)
    first_axis[..., 0] = 1
    second_axis[..., 1] = 1
    return torch.cat((rotate(quaternion, first_axis), rotate(quaternion, second_axis)), dim=-1)


def interpolate_quaternions(first, second, fraction):
    """Shortest-arc SLERP using the quaternion logarithm and exponential."""
    difference = multiply(conjugate(first), second)
    result = multiply(
        first,
        rotation_vector_to_quaternion(
            quaternion_to_rotation_vector(difference) * fraction
        ),
    )
    return result / result.norm(dim=-1, keepdim=True).clamp_min(1e-12)
