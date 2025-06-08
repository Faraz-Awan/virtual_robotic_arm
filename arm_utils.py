import numpy as np

JOINT_LIMITS = {
    "theta1": (np.radians(-135), np.radians(135)),
    "theta2": (np.radians(0), np.radians(150))
}

def forward_kinematics(theta1, theta2, L1 = 1.0, L2 = 1.0):
    """
    Compute joint and end-effector positions given joint angles.
    Returns a list of (x, y) points: base, elbow, end-effector.
    """

    x0, y0 = 0, 0 #base

    x1 = L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)

    x2 = x1 + (L2 * np.cos(theta1+theta2))
    y2 = y1 + (L2 * np.sin(theta1+theta2))

    return [(x0, y0), (x1, y1), (x2, y2)]

def inverse_kinematics(x, y, L1 = 1.0, L2 = 1.0):
    """
    Compute joint angles (theta1, theta2) for a given target point (x, y).
    Returns None if the point is unreachable.
    """

    r_squared = x**2 + y**2
    r = np.sqrt(r_squared)

    # Check reachability
    if r > (L1 + L2) or r < abs(L1 - L2):
        return None  # target is unreachable

    # Law of Cosines for theta2
    cos_theta2 = (r_squared - L1**2 - L2**2) / (2 * L1 * L2)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
    theta2 = np.arccos(cos_theta2)

    # Law of Cosines for theta1
    k1 = L1 + L2 * np.cos(theta2)
    k2 = L2 * np.sin(theta2)
    theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)

    return theta1, theta2

def within_joint_limits(theta1, theta2):
    """
    Determine joint angles (theta1, theta2) are within joint limits
    to prevent bending beyond unsafe angles.
    Returns True is joint angles are within joint limits.
    """
    t1_min, t1_max = JOINT_LIMITS["theta1"]
    t2_min, t2_max = JOINT_LIMITS["theta2"]
    return (t1_min <= theta1 <= t1_max) and (t2_min <= theta2 <= t2_max)




    