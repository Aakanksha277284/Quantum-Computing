import numpy as np

from decompose import(
    twolevel_decomposition,
    two_levels_to_unitary,
    decompose_to_basis,
    decompose_to_ht,
    circuit_to_unitary,
    error_up_to_phase,
    abc_decompose,
    abc_reconstruct,
    decompose_twolevel,
    controlled_circuit,
    gray_code,
)
# Generates a random unitary matrix
def random_unitary(n):
     
    A = np.random.randn(n,n) + 1j*np.random.randn(n,n)
    Q, R = np.linalg.qr(A)

    d = np.diag(R)
    phases = d / np.abs(d)

    return Q*phases

print("=" * 50)
print("Generating random 4X4 unitary")
print("=" * 50)

u = random_unitary(4)

print("\nUnitary Matrix:\n")
print(u)
print("\nChecking U+U = I")
print(
    np.allclose(

        u.conj().T @ u,
        np.eye(4),
        atol = 1e-8
    )
)

print ("\n" + "="*50)
print("Stage 1 Test")
print("="*50)

two_levels = twolevel_decomposition(u)
rebuilt = two_levels_to_unitary(two_levels)
err = error_up_to_phase(u, rebuilt)

#Reconstruction Error
print("Number of two-level gates:", len(two_levels))
print("Reconstruction error:", err)

#Basis Decomposition Test
print("\n" + "="*50)
print("Basis Decomposition Test")
print("="*50)

basis_circuit = decompose_to_basis(u)
rebuilt_basis = circuit_to_unitary(basis_circuit)
basis_error = error_up_to_phase(u, rebuilt_basis)

print("Gate count:" , len(basis_circuit))
print("Error:", basis_error)

#HT Decomposition Test
print("\n" + "="*50)
print("HT Decomposition Test")
print("="*50)

target_error = 0.01
ht_circuit = decompose_to_ht(u, target_error)
rebuilt_ht = circuit_to_unitary(ht_circuit)
ht_error = error_up_to_phase(u, rebuilt_ht)

print("Target error:", target_error)
print("Actual error:", ht_error)
print("Gate error:", len(ht_circuit))

print("\n" + "="*50)
print("Gate Count VS Error")
print("="*50)

errors = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3]

for e in errors:
    circuit = decompose_to_ht(u, e)
    rebuilt = circuit_to_unitary(circuit)
    actual_error = error_up_to_phase(u, rebuilt)

print(
    f"target={e:.5f}"
    f"actual={actual_error:.8f}"
    f"gates={len(circuit)}"
)

u2 = random_unitary(2)

abc = abc_decompose(u2)

u3 = abc_reconstruct(abc)

#ABC Decomposition
print(
    "ABC error =",
    error_up_to_phase(u2, u3)
)

#TwoLevel Decomposition
print("\nTesting individual two-levels\n")

tls = twolevel_decomposition(u)

for i, tl in enumerate(tls):

    circ = decompose_twolevel(tl)

    rebuilt = circuit_to_unitary(circ)

    err = error_up_to_phase(
        tl.to_unitary(),
        circuit_to_unitary(circ)
    )

    print(i, err)

    X = np.array([
    [0,1],
    [1,0]
], dtype=np.complex128)

circ = controlled_circuit(
    n=2,
    target=0,
    control_vals=[True, True],
    unitary=X
)
for tl in twolevel_decomposition(u):

    print(
        tl.level0,
        tl.level1,
        len(gray_code(tl))
    )

tl = twolevel_decomposition(u)[0]

print("TL levels:")
print(tl.level0, tl.level1)

path = gray_code(tl)


"""Conclusion"""
"""The major stages of the decomposition pipeline were implemented and tested successfully.
The two level decomposition accurately generated a random unitary matrix with recontruction error nearly 
equal to zero and the ABC decomposition error was also seen close to zero.

The number of gates increases rapidly as the error decreases because higher accuracy requires longer H/T gate 
sequences for single qubit rotations.Since a 4X4 unitary is decomposed into multiple smaller operations and each single 
qubit gate is replaced by H/T sequence, reducing the error significantly and increasing the overall gate count.

A remaining issue was in the controlled gate decomposition, which affected the final end-to-end reconstruction accuracy.
However, the result clearly show that increasing the number of gates, decreases the error aubstantially. """