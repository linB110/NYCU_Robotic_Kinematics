import numpy as np
import math as m

# reference : https://mece401.cankaya.edu.tr/uploads/files/Forward%20And%20Inverse%20Kinematics.pdf

# Conversion factor: radians → degrees
r2d = 180.0 / m.pi

# Rotation matrix about Z-axis
def _Rz(th):
    c, s = m.cos(th), m.sin(th)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], float)

# Rotation matrix about X-axis
def _Rx(th):
    c, s = m.cos(th), m.sin(th)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]], float)

# Normalize angles to (-180°, 180°]
def _wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def inverse_kinematics(kinematics_table, pos_matrix):
    """
    Compute all possible inverse kinematic solutions for the Stanford Arm.

    Parameters
    ----------
    kinematics_table : pandas.DataFrame
        DH parameter table. Must include the 'd' column containing [d1, d2, d3, d4, d5, d6].
        - d2: offset of the second joint (horizontal displacement)
        - d6: end-effector length along z6 (tool offset)
    pos_matrix : np.ndarray (4x4)
        Homogeneous transformation matrix of the end-effector (in the same length units as d2/d6).

    Returns
    -------
    np.ndarray (N x 6)
        All valid joint configurations [θ1, θ2, d3, θ4, θ5, θ6],
        where angles are in degrees and d3 keeps the same length unit as pos_matrix.
    """

    # ---- Extract DH offsets ----
    d = list(kinematics_table['d'].astype(float))
    d2 = d[1]
    d6 = d[5] if len(d) >= 6 else 0.0

    # ---- Extract rotation and translation from target pose ----
    T06 = np.asarray(pos_matrix, dtype=float)
    R06 = T06[:3, :3]
    p   = T06[:3, 3].astype(float)

    # ---- Step 1: Compute wrist center (subtract tool length d6 along z6) ----
    z6 = R06[:, 2]
    pwc = p - d6 * z6
    px, py, pz = float(pwc[0]), float(pwc[1]), float(pwc[2])

    # ---- Step 2: Solve shoulder angle θ1 (two solutions: left/right) ----
    r = m.hypot(px, py)
    if r + 1e-12 < abs(d2):
        raise ValueError("No valid solution: sqrt(px^2 + py^2) < |d2|")

    delta = m.atan2(-px, py)
    gamma = m.acos(max(-1.0, min(1.0, d2 / max(r, 1e-15))))
    theta1_set = [delta + gamma, delta - gamma]  # Shoulder left/right

    sols = []

    # ---- Step 3: For each θ1, solve θ2 and d3 ----
    for th1 in theta1_set:
        c1, s1 = m.cos(th1), m.sin(th1)
        u = px * c1 + py * s1

        # Compute both θ2 definitions (sign ambiguity in geometry)
        theta2_bases = [m.atan2(u, pz), m.atan2(-u, pz)]
        R01 = _Rz(th1) @ _Rx(-m.pi/2)

        best_pairs = []
        for th2_base in theta2_bases:
            d3_abs = m.hypot(u, pz)
            # Elbow up/down pair: (θ2, +d3) and (θ2+π, -d3)
            candidates = [(th2_base, +d3_abs), (th2_base + m.pi, -d3_abs)]

            # Geometric test: choose the pair minimizing position error
            z1 = np.array([-s1, c1, 0.0])  # z1(θ1)
            def z2(t2):
                c2, s2 = m.cos(t2), m.sin(t2)
                return np.array([c1 * s2, s1 * s2, c2])

            best = None
            for t2, d3v in candidates:
                pwc_hat = d2 * z1 + d3v * z2(t2)
                err = float(np.linalg.norm(pwc_hat - pwc))
                if (best is None) or (err < best[0]):
                    best = (err, t2, d3v)
            best_pairs.append(best)  # keep the smaller-error pair for this θ2 base

        # ---- Step 4: For each valid (θ1, θ2, d3), solve wrist angles θ4, θ5, θ6 ----
        for _, th2, d3v in best_pairs:
            R12 = _Rz(th2) @ _Rx(+m.pi/2)
            R03 = R01 @ R12
            R36 = R03.T @ R06

            # Wrist angles (two configurations for θ5 ≠ 0, π)
            c5 = max(-1.0, min(1.0, float(R36[2, 2])))
            th5 = m.acos(c5)
            s5 = m.sin(th5)

            if abs(s5) > 1e-10:
                th4a = m.atan2(R36[1, 2], R36[0, 2])
                th6a = m.atan2(R36[2, 1], -R36[2, 0])
                th4b = m.atan2(-R36[1, 2], -R36[0, 2])
                th6b = m.atan2(-R36[2, 1],  R36[2, 0])
                wrists = [(th4a, +th5, th6a),
                          (th4b, -th5, th6b)]
            else:
                # Wrist singularity: θ5 ≈ 0 or π → infinite θ4/θ6 combinations
                yaw = m.atan2(-R36[0, 1], R36[0, 0])
                wrists = [(0.0, th5, yaw),
                          (m.pi, th5, yaw + m.pi)]

            # ---- Collect all valid 8 solutions ----
            for th4, th5v, th6 in wrists:
                sols.append([
                    _wrap_deg(th1 * r2d),
                    _wrap_deg(th2 * r2d),
                    d3v,
                    _wrap_deg(th4 * r2d),
                    _wrap_deg(th5v * r2d),
                    _wrap_deg(th6 * r2d),
                ])

    # ---- Step 5: Remove duplicates (handle numeric noise or singularities) ----
    uniq, seen = [], set()
    for s in sols:
        key = (round(s[0], 6), round(s[1], 6), round(s[2], 9),
               round(s[3], 6), round(s[4], 6), round(s[5], 6))
        if key not in seen:
            seen.add(key)
            uniq.append(s)

    # Return as numpy array, shape (N, 6)
    return np.array(uniq, dtype=float)

