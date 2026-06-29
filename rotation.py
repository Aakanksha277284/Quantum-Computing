import numpy as np

# Use a single complex dtype for numpy everywhere.
DTYPE = np.complex128

INV_SQRT2 = 1.0 / np.sqrt(2.0)
H = INV_SQRT2 * np.array([[1, 1], [1, -1]], dtype=DTYPE)

# LAMBDA_PI is the base rotation angle realized by the H/T building blocks:
# cos(LAMBDA_PI) = cos^2(pi/8) = (1 + 1/sqrt2)/2. Because LAMBDA_PI / (2 pi) is
# irrational, the multiples {k * LAMBDA_PI mod 2 pi} densely fill [0, 2 pi).
LAMBDA_PI = np.arccos((1.0 + INV_SQRT2) / 2.0)
TWO_PI = 2.0 * np.pi


class Bloch:
    """Axis-angle (Bloch) form of a 2x2 unitary G:

        G = e^{i alpha} (cos(theta/2) I - i sin(theta/2) (n . sigma))

    i.e. a global phase e^{i alpha} times a rotation by angle `theta` about the
    Bloch-sphere axis `n`. Here (n . sigma) = n_x X + n_y Y + n_z Z.
    """

    alpha: float  # global phase
    n: np.ndarray  # unit rotation axis, shape (3,): [n_x, n_y, n_z]
    theta: float  # rotation angle


def to_bloch(g: np.ndarray) -> Bloch:
    """Recover the Bloch form (alpha, n, theta) of a 2x2 unitary `g`."""
    #Global phase
    det = np.linalg.det(g)
    alpha = np.angle(det) / 2.0

    # Remove gkobal phase
    u = g * np.exp(-1j*alpha)
 
    # Rotation angle
    trace = np.trace(u)
    cos_half_theta = np.real(trace)/ 2.0
    cos_half_theta = np.clip(cos_half_theta, -1.0, 1.0)

    theta = 2.0 * np.arccos(cos_half_theta)

    # Rotation axis
    if np.isclose(theta, 0):
        n = np.array([1.0,0.0,0.0])
    else:
        s = np.sin(theta / 2.0)

        a, b = u[0,0], u[0,1]
        c, d = u[1,0], u[1,1]

        nx = np.imag(b+c) / (2*s)
        ny = np.real(c-b) / (2*s)
        nz = np.imag(d-a) / (2*s)

        n = np.array([nx,ny,nz], dtype=float)
        n = n / np.linalg.norm(n)

    return Bloch(alpha, n, theta)
  # raise NotImplementedError("to_bloch is not implemented yet")


# n1, n2 are two orthogonal Bloch-sphere axes (n1 . n2 == 0)
# TODO: fill in the two orthogonal rotation axes (each a length-3
# unit vector [x, y, z])
#Orthogonal axes (X,Y)
n1 = np.array([1.0,0.0,0.0])
n2 = np.array([0.0,1.0,0.0])

# frame derived from the axes (given)
# take the dot product of the Bloch axis with these
# the minus sign arises from the double cover issue
a1 = -n1
a2 = -n2
a3 = np.cross(a1, a2)

def rotation_matrix(n, theta):
    nx, ny, nz = n
    c = np.cos(theta)
    s = np.sin(theta)
    v = 1 - c

    return np. array([
        [c + nx*nx*v, nx*ny*v - nz*s, nx*nz*v + ny*s],
        [ny*nx*v + nz*s, c + ny*ny*v, ny*nz*v - nx*s],
        [nz*nx*v - ny*s, nz*ny*v + nx*s, c + nz*nz*v]
    ])

def n1n2n1_angles(b: Bloch) -> tuple[float, float, float, float]:
    """Factor the rotation part of a unitary (given as its Bloch form `b`) as
        u = e^{i global_phase} * Rn1(alpha) * Rn2(beta) * Rn1(gamma)

    where Ra(angle) is a rotation by `angle` about axis a, and {a1, a2, a3} is
    the orthonormal frame defined above. Returns (alpha, beta, gamma, global_phase).
    """

    R = rotation_matrix(b.n, b.theta)
    #X-Y-X Decomposition
    beta = np.arccos(np.clip(R[0,0], -1.0, 1.0))

    if np.isclose(np.sin(beta), 0):
        alpha = 0.0
        gamma = np.arctan2(-R[1,2], R[1,1])
    else:
        alpha = np.arctan2(R[1,0], -R[2,0])
        gamma = np.arctan2(R[0,1], R[0,2])

    return alpha % TWO_PI, beta % TWO_PI, gamma % TWO_PI, b.alpha

    # TODO(student): implement using the steps above.
    raise NotImplementedError("n1n2n1_angles is not implemented yet")

def wrap_angle(x):
    return x % TWO_PI

def angle_dist(a,b):
    d = abs(a-b)
    return min(d, TWO_PI - d)

def approx_angle_with_tolerance(angle: float, tolerance: float) -> int:
    """Find an integer multiple k such that
        (k * LAMBDA_PI) mod 2*pi  ~=  angle   (within `tolerance`)
    Since LAMBDA_PI / (2 pi) is irrational, such a k always exists; search
    k = 1, 2, 3, ... and return the first one whose wrapped multiple lands within
    `tolerance` of `angle` (compare both as angles in [0, 2 pi)).

    Hint:
      * wrap an angle into [0, 2 pi)
      * the angular distance between two wrapped angles a, b is
        min(|a - b|, TWO_PI - |a - b|) (so 0.01 and 2*pi - 0.01 count as close).
    """
    target = wrap_angle(angle)

    for k in range(100000):
        val = wrap_angle(k * LAMBDA_PI)
        if angle_dist(val, target) <= tolerance:
            return k
        
    raise RuntimeError("No approximation found within search range")

    # TODO(student): implement using the hint above.
    raise NotImplementedError("approx_angle_with_tolerance is not implemented yet")


def decompose_2x2(u: np.ndarray, tolerance: float) -> tuple[int, int, int]:
    """Approximate a 2x2 unitary `u` as a product of powers of M1 and M2:

        u  ~=  M1^k * M2^l * M1^m     (up to a global phase)

    where M1 is a rotation about axis a1 and M2 a rotation about axis a2, each by
    the base angle realized by the H/T building blocks. Returns the powers
    (k, l, m).

    Steps (combine the two functions above):

      1. Get the Bloch form of u (to_bloch), then factor its rotation into the
         three frame angles with n1n2n1_angles:
             alpha, beta, gamma, _global_phase = n1n2n1_angles(to_bloch(u))
         alpha and gamma are rotations about a1 (realized by powers of M1);
         beta is a rotation about a2 (realized by powers of M2).

      2. Convert each angle to an integer power with approx_angle_with_tolerance:
             k = approx_angle_with_tolerance(alpha, tolerance)   # power of M1
             l = approx_angle_with_tolerance(beta,  tolerance)   # power of M2
             m = approx_angle_with_tolerance(gamma, tolerance)   # power of M1
         (Mind the relationship between a target rotation angle and the base
         angle each application of M1/M2 adds.)

      3. Return (k, l, m).
    """

    b = to_bloch(u)
    alpha, beta, gamma, _ = n1n2n1_angles(b)

    k = approx_angle_with_tolerance(alpha, tolerance)
    l = approx_angle_with_tolerance(beta, tolerance)
    m = approx_angle_with_tolerance(gamma, tolerance)

    return k,l,m
    # TODO(student): implement using the steps above.
    raise NotImplementedError("decompose_2x2 is not implemented yet")
