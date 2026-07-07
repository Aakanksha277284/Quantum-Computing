from dataclasses import dataclass

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

@dataclass
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
n1 = np.array([0.0,0.0,1.0])
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
    beta = np.arccos(np.clip(R[2,2], -1.0, 1.0))

    if abs(np.sin(beta)) < 1e-10:
        alpha = 0.0
        gamma = np.arctan2(-R[1,0], R[0,0])
    else:
        alpha = np.arctan2(R[1,2], R[0,2])
        gamma = np.arctan2(R[2,1], -R[2,0])

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


# ---------------------------------------------------------------------------
# Single-qubit rotation helpers (see cpp/src/Unitary2_Bloch.h).
#
# These are the inverse/companion operations to to_bloch and are reused by the
# multi-qubit decomposition pipeline in decompose.py.
# ---------------------------------------------------------------------------


def from_axis_angle(b: Bloch) -> np.ndarray:
    """Build a 2x2 unitary from its Bloch form: a global phase times a rotation
    by angle b.theta about axis b.n (inverse of to_bloch).

        G = e^{i b.alpha} (cos(b.theta/2) I - i sin(b.theta/2) (b.n . sigma))

    where (b.n . sigma) = n_x X + n_y Y + n_z Z. Assumes b.n is a unit vector.
    """
    nx, ny, nz = b.n

    X = np.array([[0,1],[1,0]], dtype = DTYPE)
    Y = np.array([[0,-1j],[1j,0]], dtype = DTYPE)
    Z = np.array([[1,0],[0,-1]], dtype = DTYPE)

    c = np.cos(b.theta/2.0)
    s = np.sin(b.theta/2.0)

    n_sigma = nx*X + ny*Y + nz*Z
    rotation = c*np.eye(2, dtype = DTYPE) - 1j*s*n_sigma
    return np.exp(1j*b.alpha)*rotation

    # TODO: implement using the formula above.
    raise NotImplementedError("from_axis_angle is not implemented yet")


def Rz(theta: float) -> np.ndarray:
    """Rotation about the z axis (no global phase):

    Rz(theta) = diag(e^{-i theta/2}, e^{i theta/2}).
    """
    return from_axis_angle(
        Bloch(
            0.0,
            np.array([0.0,0.0,1.0]),
            theta
        )
    )
    # TODO: implement (hint: from_axis_angle about axis [0, 0, 1]).
    raise NotImplementedError("Rz is not implemented yet")


def Ry(theta: float) -> np.ndarray:
    """Rotation about the y axis (no global phase):

    Ry(theta) = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]].
    """

    return from_axis_angle(
        Bloch(
            0.0,
            np.array([0.0,1.0,0.0]),
            theta
        )
    )
    # TODO: implement (hint: from_axis_angle about axis [0, 1, 0]).
    raise NotImplementedError("Ry is not implemented yet")


def euler_angles_zyz(u: np.ndarray) -> tuple[float, float, float, float]:
    """ZYZ Euler decomposition of a 2x2 unitary: angles (alpha, beta, gamma, delta)
    with

        u = e^{i alpha} Rz(beta) Ry(gamma) Rz(delta).

    alpha is the global phase (arg(det u)/2); the rest come from S = e^{-i alpha} u
    in SU(2), where s00 = cos(gamma/2) e^{-i(beta+delta)/2} and
    s10 = sin(gamma/2) e^{i(beta-delta)/2}. When gamma = 0 (s10 = 0), beta/delta are
    split arbitrarily (gimbal lock); the identity still holds.
    """
    alpha = np.angle(np.linalg.det(u))/ 2.0
    s = u * np.exp(-1j*alpha)

    s00 = s[0,0]
    s10 = s[1,0]

    gamma = 2.0*np.arctan2(abs(s10), abs(s00))

    if np.isclose(abs(s10), 0.0):
        beta = 0.0
        delta = -2.0*np.angle(s00)
    else:
        p = np.angle(s10)
        q = np.angle(s00)

        beta = p-q
        delta = -(p+q)

    return (alpha % TWO_PI, beta % TWO_PI, gamma % TWO_PI, delta % TWO_PI, )
    # TODO: implement using the relations above.
    raise NotImplementedError("euler_angles_zyz is not implemented yet")


def unitary2_sqrt(u: np.ndarray) -> np.ndarray:
    """Principal square root: a 2x2 unitary V with V @ V == u, phase included.
    Take the Bloch form of u and halve both alpha and theta (same axis); squaring
    back doubles them, reproducing u exactly.
    """
    b = to_bloch(u)
    return from_axis_angle(
        Bloch(
            b.alpha/2.0,
            b.n,
            b.theta/2.0
        )
    )

    # TODO: implement (hint: to_bloch, halve alpha and theta, from_axis_angle).
    raise NotImplementedError("unitary2_sqrt is not implemented yet")


# ---------------------------------------------------------------------------
# H/T word machinery for approximating a 2x2 unitary in {H, T} (see cpp/src/HT.h).
#
# M1, M2 are short H/T words that realize rotations by THETA_M = 2*LAMBDA_PI about
# the axes a1, a2. A word is a flat string of 'H'/'T' characters, read left-to-right
# as a matrix product (leftmost char = leftmost/outermost factor).
# ---------------------------------------------------------------------------

# alternating (T-power, H-power, ...) exponents, starting with T
M1_WORD = [7, 1, 1, 1]
M2_WORD = [2, 1, 1, 1, 6, 1, 7, 1, 5, 1, 1, 1, 2, 1, 1, 1, 2, 1, 7, 1, 6]


def expand_word(word: list[int]) -> str:
    """Flatten an alternating (T-power, H-power, ...) exponent list into a literal
    string of 'H'/'T' gates (left-to-right). Even indices are T, odd indices are H.
    """
    pieces = []

    for i, count in enumerate(word):
        if i%2 == 0:
            pieces.append("T"*count)
        else:
            pieces.append("H"*count)
    
    return " ".join(pieces)
    # TODO: implement.
    raise NotImplementedError("expand_word is not implemented yet")


# flat H/T strings for the two building-block words (computed once expand_word works)
M1_STR = expand_word(M1_WORD)
M2_STR = expand_word(M2_WORD)


def gates_to_unitary(gates: str) -> np.ndarray:
    """The 2x2 unitary of a flat H/T gate string (left-to-right product)."""
    T = np.array([[1,0], [0, np.exp(1j*np.pi/4)]], dtype = DTYPE)

    u = np.eye(2, dtype = DTYPE)
    for gate in gates:
        if gate == "H":
            u = H @ u
        elif gate == "T":
            u = T @ u
        else:
            raise ValueError(f"Unkown gate: {gate}")
        
    return u
    # TODO: implement (multiply H / T for each char, starting from I).
    raise NotImplementedError("gates_to_unitary is not implemented yet")


def invert_gates(gates: str) -> str:
    """Inverse of a flat H/T word: reverse the gate order and invert each gate.
    H^-1 = H; the {H, T} basis has no T-dagger, so T^-1 must be spelled as T^7.
    """
    inverse = []
    for gate in reversed(gates):
        if gate == "H":
            inverse.append("H")
        elif gate == "T":
            inverse.append("T"*7)
    
    return "".join(inverse)
    # TODO: implement.
    raise NotImplementedError("invert_gates is not implemented yet")


def power_gates(base: str, k: int) -> str:
    """The k-th power of a flat H/T word: base repeated k times. Negative k uses the
    inverse word (invert_gates).
    """
    if k == 0:
     return ""
    
    if k > 0:
       return base*k
   
    return invert_gates(base)*(-k)
    # TODO: implement.
    raise NotImplementedError("power_gates is not implemented yet")


def approximate_in_ht(u: np.ndarray, error: float) -> str:
    """Approximate a 2x2 unitary `u` by a flat H/T word (up to global phase) to the
    angular tolerance `error` (smaller -> longer, more accurate).

    Use decompose_2x2 to get the powers (k, l, m) with u ~= M1^k M2^l M1^m, then
    assemble the word:

        power_gates(M1_STR, k) + power_gates(M2_STR, l) + power_gates(M1_STR, m).
    """
    k,l,m = decompose_2x2(u, error)

    p1 = power_gates(M1_STR, k)
    p2 = power_gates(M2_STR, l)
    p3 = power_gates(M1_STR, m)

    return p1+p2+p3
    # TODO: implement using decompose_2x2 and power_gates.
    raise NotImplementedError("approximate_in_ht is not implemented yet")
