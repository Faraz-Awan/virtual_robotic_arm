import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def forward_kinematics(theta1, theta2, L1=1.0, L2=1.0):
    x0, y0 = 0, 0
    x1 = x0 + L1 * np.cos(theta1)
    y1 = y0 + L1 * np.sin(theta1)
    x2 = x1 + L2 * np.cos(theta1 + theta2)
    y2 = y1 + L2 * np.sin(theta1 + theta2)
    return [(x0, y0), (x1, y1), (x2, y2)]

def inverse_kinematics(x, y, L1=1.0, L2=1.0):
    r = np.sqrt(x**2 + y**2)
    if r > L1 + L2:
        return None

    cos_theta2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    sin_theta2 = np.sqrt(1 - cos_theta2**2)
    theta2 = np.arctan2(sin_theta2, cos_theta2)

    k1 = L1 + L2 * cos_theta2
    k2 = L2 * sin_theta2
    theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)
    return theta1, theta2

# --- Streamlit UI ---
st.title("🦾 2DOF Robotic Arm Simulator")

st.markdown("Click anywhere in the workspace to set a target point.")

x = st.slider("Target X", -2.0, 2.0, 1.0)
y = st.slider("Target Y", -2.0, 2.0, 1.0)

result = inverse_kinematics(x, y)

fig, ax = plt.subplots()
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal')
ax.grid(True)

if result:
    theta1, theta2 = result
    points = forward_kinematics(theta1, theta2)
    xs, ys = zip(*points)
    ax.plot(xs, ys, marker='o', linewidth=3, markersize=8)
    ax.set_title(f"Target: ({x:.2f}, {y:.2f})")
else:
    ax.set_title("Target unreachable")

st.pyplot(fig)