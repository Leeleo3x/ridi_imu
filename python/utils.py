import quaternion
import numpy as np


def quaternion_from_two_vectors(v1, v2):
    q = quaternion.quaternion(1, 0, 0, 0)
    q.vec = np.cross(v1, v2)
    q.w = np.sqrt((np.linalg.norm(v1) * np.linalg.norm(v2))) + np.dot(v1, v2)
    return q
