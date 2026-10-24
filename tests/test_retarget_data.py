"""Geometric checks for the independent direct BVH pipeline; no simulator needed."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_processing"))
from bvh import read_bvh
from bvh import rigid_torso_frame


class BVHGeometryTests(unittest.TestCase):
    def test_rigid_torso_uses_landmark_chord_and_is_rotation_equivariant(self):
        torso_rotation = Rotation.from_euler("y", 25, degrees=True).as_matrix()
        hips = np.zeros((1, 3))
        neck = (torso_rotation @ np.array([0, 0, 0.6]))[None]
        left = neck + (torso_rotation @ np.array([0, 0.2, 0]))[None]
        right = neck + (torso_rotation @ np.array([0, -0.2, 0]))[None]
        frame = rigid_torso_frame(hips, neck, left, right)
        np.testing.assert_allclose(frame[0], torso_rotation, atol=1e-12)
        world_rotation = Rotation.from_euler("xyz", [10, -15, 60], degrees=True).as_matrix()
        translation = np.array([2, -3, 5])
        moved = [points @ world_rotation.T + translation for points in (hips, neck, left, right)]
        np.testing.assert_allclose(rigid_torso_frame(*moved)[0], world_rotation @ torso_rotation, atol=1e-12)

    def test_declared_rotation_order_translation_and_world_basis(self):
        text = """HIERARCHY
ROOT Hips
{
 OFFSET 999 888 777
 CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
 JOINT Child
 {
  OFFSET 100 0 0
  CHANNELS 3 Yrotation Zrotation Xrotation
  End Site
  {
   OFFSET 0 0 0
  }
 }
}
MOTION
Frames: 2
Frame Time: 0.033333
100 200 300 90 0 0 0 0 0
200 300 400 30 20 10 40 50 60
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "geometry.bvh"
            path.write_text(text)
            motion = read_bvh(path)
        basis = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        np.testing.assert_allclose(motion.positions[0, 0], [1, -3, 2], atol=1e-12)
        np.testing.assert_allclose(motion.positions[0, 1], [1, -3, 3], atol=1e-12)
        expected_root = basis @ Rotation.from_euler("ZXY", [30, 20, 10], degrees=True).as_matrix()
        expected_child = expected_root @ Rotation.from_euler("YZX", [40, 50, 60], degrees=True).as_matrix()
        np.testing.assert_allclose(motion.rotations[1, 1], expected_child, atol=1e-12)
        np.testing.assert_allclose(motion.positions[1, 1] - motion.positions[1, 0], expected_root[:, 0], atol=1e-12)
        self.assertEqual(motion.parents, [-1, 0])


if __name__ == "__main__":
    unittest.main()
