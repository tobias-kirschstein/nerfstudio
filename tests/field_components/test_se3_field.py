import logging
from statistics import mean
from unittest import TestCase
import tinycudann as tcnn

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from nerfstudio.fields.warping import skew, exp_so3, exp_se3, SE3Field
from scipy.spatial.transform.rotation import Rotation as R


class TestSE3Field(TestCase):

    def test_skew(self):
        B = 7
        w = torch.randn((B, 3))
        W = skew(w)
        v = torch.randn((B, 3, 1))

        result = (W @ v).squeeze(2)  # [B, 3]
        result_ref = torch.linalg.cross(w, v.squeeze(), dim=1)  # [B, 3]
        self.assertTrue((result == result_ref).all())

    def test_exp_so3(self):
        B = 7
        w = torch.randn((B, 3))
        theta = w.norm(dim=1, p=2).unsqueeze(1)
        w /= theta
        W = skew(w)
        R1 = exp_so3(W, theta).numpy()  # [B, 3, 3]

        # rotvecs = w / w.norm(dim=1, p=2).unsqueeze(1)
        rotvecs = w * theta
        R2 = R.from_rotvec(rotvecs).as_matrix()

        self.assertEqual(len(R1.shape), 3)
        self.assertEqual(R1.shape[0], B)
        self.assertEqual(R1.shape[1], 3)
        self.assertEqual(R1.shape[2], 3)
        self.assertTrue(np.all(np.abs(R1 - R2) < 1e-6))

    def test_exp_se3(self):
        B = 7
        s = torch.randn((1, 6)).repeat((B, 1))  # screw axes. Using the same across batch
        theta = s[:, :3].norm(dim=1, p=2).unsqueeze(1)
        s /= theta  # factor out theta from rotation and translation vectors. rotation vector will be unit vector
        theta = torch.arange(B).reshape(
            (B, 1)) * 2 * np.pi  # Only use full rotations for testing (basically is only translation then)

        poses = exp_se3(s, theta)

        rotvecs = s[:, :3] * theta
        R1 = R.from_rotvec(rotvecs).as_matrix()
        t = s[:, 3:6]
        poses_2 = np.concatenate([R1, t.unsqueeze(2).numpy()], axis=2)

        for pose_1, pose_2, pose_3 in zip(poses[:-2], poses[1:-1], poses[2:]):
            # screw motion is "evaluated" at full 2pi rotations => it is a linear movement
            # Points should move out linearly. I.e., the distance between subsequent points remains constant
            t_diff_1 = pose_2[:3, 3] - pose_1[:3, 3]
            t_diff_2 = pose_3[:3, 3] - pose_2[:3, 3]
            self.assertLess((t_diff_1 - t_diff_2).norm(), 1e-5)
        self.assertEqual(len(poses.shape), 3)
        self.assertEqual(poses.shape[0], B)
        self.assertEqual(poses.shape[1], 4)
        self.assertEqual(poses.shape[2], 4)

    def test_train_se3_field_direct_simple(self):
        T = 8
        D = 16
        V = 4

        se3_field = SE3Field(D, 16)
        deformation_field = tcnn.Network(3 + D, 3, network_config={
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 16,
            "n_hidden_layers": 3,
        })

        time_embedding = nn.Embedding(T, D)
        time_embedding = time_embedding.cuda()
        loss_fn = F.l1_loss
        optimizer = torch.optim.Adam([*se3_field.parameters(), *time_embedding.parameters()], lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=8e-2)
        # optimizer = torch.optim.Adam([*deformation_field.parameters(), *time_embedding.parameters()], lr=1e-2)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=8e-2)

        points = torch.meshgrid([torch.arange(V), torch.arange(V), torch.arange(V)])
        points = torch.stack(points, dim=-1).view(-1, 3)
        points = points.cuda().float()

        B = points.shape[0]
        mean_loss = 100
        for i in range(500):
            losses = []
            for t in range(T):
                targets = points + t
                embeddings = time_embedding(torch.tensor(t).repeat(B).cuda())

                # predictions = deformation_field(torch.concat([points, embeddings], dim=1))

                predictions = se3_field(points=points, latent_codes=embeddings)
                loss = loss_fn(predictions, targets)
                loss.backward()
                losses.append(loss.item())
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            mean_loss = mean(losses)
            print(f"{mean_loss:0.4f}")

        self.assertLess(mean_loss, 0.01)

    def test_train_se3_field_direct(self):
        # TODO: Why do the deformation field and the SE3Field have such a hard time fitting this simple example?
        #   Normalization of input coordinates helped, but why does it matter?

        T = 8
        D = 16
        H = 128
        V = 4
        BATCH_SIZE = 256
        movement_magnitude = 1

        se3_field = SE3Field(D, H, use_bias=True, n_frequencies=12)
        deformation_field = tcnn.Network(3 + D, 3, network_config={
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": H,
            "n_hidden_layers": 5,
        })

        time_embedding = nn.Embedding(T, D)
        time_embedding = time_embedding.cuda()
        torch.nn.init.normal_(time_embedding.weight, mean=0, std=0.01)
        loss_fn = F.l1_loss
        # optimizer = torch.optim.Adam([*se3_field.parameters(), *time_embedding.parameters()], lr=5e-3)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=8e-2)
        optimizer = torch.optim.Adam([*deformation_field.parameters(), *time_embedding.parameters()], lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=1e-1)

        # linear box movement
        movement = torch.zeros((V, V, V))
        movement[int(V / 4):int(3 / 4 * V), int(V / 4):int(3 / 4 * V), int(V / 4):int(3 / 4 * V)] = movement_magnitude
        # movement[:] = movement_magnitude
        movement = movement.cuda()

        points = torch.meshgrid([torch.arange(V), torch.arange(V), torch.arange(V)])
        points = torch.stack(points, dim=-1).view(-1, 3)
        points = (points - V / 2) / (V / 2)  # Normalization is important, otherwise optimization does not work well
        points = points.cuda().float()

        B = points.shape[0]
        dataset_points = []
        dataset_times = []
        dataset_targets = []
        for t in range(T):
            targets = points + torch.tensor(t).repeat(B).cuda().unsqueeze(-1) * movement.view(-1, 1)
            dataset_points.append(points)
            dataset_times.append(torch.tensor(t).repeat(B).cuda())
            dataset_targets.append(targets)

        dataset_points = torch.concat(dataset_points)
        dataset_times = torch.concat(dataset_times)
        dataset_targets = torch.concat(dataset_targets)

        for i in range(5000):
            losses = []
            # idx = torch.randint(len(dataset_points), (BATCH_SIZE,))
            # ps = dataset_points[idx]
            # times = dataset_times[idx]
            # targets = dataset_targets[idx]
            # embeddings = time_embedding(times)
            # predictions = se3_field(points=ps, latent_codes=embeddings)
            # loss = loss_fn(predictions, targets)
            # loss.backward()
            # losses.append(loss.item())

            for t in range(T):
                targets = points + torch.tensor(t / V).repeat(B).cuda().unsqueeze(-1) * movement.view(-1, 1)
                embeddings = time_embedding(torch.tensor(t).repeat(B).cuda())

                predictions = deformation_field(torch.concat([points, embeddings], dim=1)).to(points)

                # predictions = se3_field(points=points, latent_codes=embeddings)
                loss = loss_fn(predictions, targets)
                loss.backward()
                losses.append(loss.item())
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            mean_loss = mean(losses)
            print(f"{mean_loss:0.4f}")

        self.assertLess(mean_loss, 0.01)
