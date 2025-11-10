r"""Nonlinear Transformation of Complex Amplitudes via Quantum Singular Value Transformation
========================================================================================

Quantum mechanics is inherently linear: unitary operations act linearly on state amplitudes, and
measurement is the only source of nonlinearity. Nevertheless, many useful algorithms – such as
neural networks – rely on nonlinear functions. The recent work by Guo, Mitarai and Fujii (2024) [#nlat]_
introduces **nonlinear transformation of complex amplitudes (NTCA)**, a task where a quantum circuit
transforms the amplitudes :math:`c_k = x_k + i y_k` of a state :math:`\sum_k c_k\,|k\rangle` into
new amplitudes :math:`P(x_k)+Q(y_k)` defined by real‐valued functions :math:`P, Q`. The authors show
how to achieve NTCA using a combination of *block‐encoding of amplitudes* and **quantum singular
value transformation (QSVT)**, exploiting post‐selection to realise a nonlinear map while
maintaining overall unitary evolution.

In this demo we first explain how amplitudes are embedded into a Hermitian block so that their real
and imaginary parts can be accessed, and then briefly review QSVT. The second part uses this
machinery to build a simple quantum classifier that acts nonlinearly on encoded data via NTCA.
"""

######################################################################
# Toy dataset and linear-classifier baseline
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# **Data shape.** Inputs :math:`x \in [-1,1]`. Labels :math:`y=1` when :math:`|x|>0.4`, otherwise
# :math:`y=0`. Thus the positive class consists of two disjoint intervals
# :math:`[-1,-0.4)\cup(0.4,1]`.
# 
# **Why linear fails.** A one–dimensional linear classifier with a monotonic link (e.g. logistic
# :math:`\sigma(ax+b)`) can implement only a single threshold. Its decision region for class :math:`1`
# is a connected interval of the form :math:`(-\infty,t)` or :math:`(t,\infty)`. Our target region is
# disconnected, so no single linear threshold can capture both positive lobes.
# 
# **Empirical confirmation.** We search the midpoints between our sample points for the best single
# threshold, trying both orientations. Even the optimal 1D threshold misclassifies several of the
# twenty samples. The mismatch is geometric, not due to poor parameter tuning.
# 
# **Where nonlinearity enters the chat.** Introducing a nonlinear feature map :math:`f(x)` or a
# saturating activation on quantum amplitudes allows us to carve out multiple intervals. In the
# NTCA/QSVT construction, a two–block circuit with nonlinear activations acts like two concatenated
# thresholds, perfectly fitting this dataset.
# 
# **Narrative fit.** The two subplots below illustrate the need for nonlinearity. The first shows the
# target labels and boundaries at :math:`\pm 0.4`, and the second shows the best linear–threshold
# classifier’s predictions and misclassifications. This sets the stage for using NTCA, which
# constructs physically meaningful nonlinearities to implement the desired two–interval decision
# boundary.
# 

# Visualize the toy dataset and the best linear-threshold classifier
import numpy as np
import matplotlib.pyplot as plt

# Generate toy dataset
X_demo = np.linspace(-1.0, 1.0, 20)
y_demo = np.array([1 if abs(xi) > 0.4 else 0 for xi in X_demo])

# Compute the best linear threshold (single cut) and orientation
thresholds = [(X_demo[i] + X_demo[i+1]) / 2 for i in range(len(X_demo) - 1)]
best_t = None
best_orient = None
best_mis = len(X_demo)
for t in thresholds:
    for orient in [1, -1]:
        preds = (orient * (X_demo - t) > 0).astype(int)
        mis = np.sum(preds != y_demo)
        if mis < best_mis:
            best_mis = mis
            best_t = t
            best_orient = orient

# Predictions using the best threshold
best_preds = (best_orient * (X_demo - best_t) > 0).astype(int)

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)

# Top plot: toy dataset with boundaries
axs[0].scatter(X_demo[y_demo==0], y_demo[y_demo==0], color='tab:blue', label='class 0', s=80)
axs[0].scatter(X_demo[y_demo==1], y_demo[y_demo==1], color='tab:orange', label='class 1', s=80)
axs[0].axvline(-0.4, color='goldenrod', linestyle='--', linewidth=1, label='boundary -0.4')
axs[0].axvline(0.4, color='peru', linestyle='--', linewidth=1, label='boundary 0.4')
axs[0].set_xlabel('x')
axs[0].set_ylabel('class')
axs[0].set_title('Toy dataset for the NTCA quickstart')
axs[0].set_yticks([0, 1])
axs[0].set_yticklabels(['class 0', 'class 1'])
axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

# Bottom plot: best linear-threshold prediction
axs[1].scatter(X_demo[y_demo==0], y_demo[y_demo==0], color='tab:blue', label='class 0 (data)', s=80)
axs[1].scatter(X_demo[y_demo==1], y_demo[y_demo==1], color='tab:orange', label='class 1 (data)', s=80)
axs[1].step(X_demo, best_preds, where='mid', color='darkorange', label='best linear-threshold prediction')
# Mark chosen threshold line
axs[1].axvline(best_t, color='goldenrod', linestyle='--', linewidth=1, label=f'chosen threshold t={best_t:.3f}')
axs[1].set_xlabel('x')
axs[1].set_ylabel('class')
axs[1].set_title(f'Best 1D linear-threshold fit: {best_mis} misclassified out of {len(X_demo)}')
axs[1].set_yticks([0,1])
axs[1].set_yticklabels(['class 0','class 1'])
axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)

plt.show()


######################################################################
# 1. Block‑encoding of amplitudes
# -------------------------------
# 
# To implement nonlinear functions of amplitudes, we first need to *extract* the real and imaginary
# parts of :math:`c_k` into the spectrum of a Hermitian operator. Suppose we have a state preparation
# oracle
# 
# .. math::
# 
# 
#    U:|0\rangle \longmapsto \sum_{k=1}^N c_k\,|k\rangle.
# 
# The goal is to construct a Hermitian matrix whose eigenvalues are :math:`\{x_k\}` or
# :math:`\{y_k\}`, where :math:`c_k=x_k+ i y_k`. [#nlat]_ shows that this can be done with a
# *block‑encoding* :math:`\widetilde G` built from :math:`U` and its adjoint: the operator
# :math:`\widetilde G` acts on an expanded register of :math:`(2n+1)` qubits and satisfies
# 
# .. math::
# 
#    (\langle 0| \otimes I_{2n+1})\, \widetilde G\,(|0\rangle\otimes I_{2n+1}) = \sum_{k=1}^N x_k\,|\phi_k\rangle\langle\phi_k| + \cdots,
# 
# and similarly for :math:`\widetilde G'` encoding :math:`y_k`. The construction uses the following
# ingredients:
# 
# - A *coherent walk* operator :math:`W` based on the state preparation unitary :math:`U`. In essence,
#   :math:`W` prepares a superposition of the prepared state and the basis state :math:`|k\rangle`,
#   storing the sum and difference in an ancilla qubit :math:`B`.
# - A conditional phase shift :math:`S_0=I_{n+1}-2|0\rangle\langle 0|` acting on the data qubits and
#   ancilla :math:`B`, and a Pauli–Z on :math:`B`.
# - Combining these operations to form :math:`G := W S_0 W^\dagger Z_B`. The eigenvalues of :math:`G`
#   have real part :math:`-x_k` and imaginary part :math:`\pm \sqrt{1-x_k^2}`. Taking the Hermitian
#   part
# 
#   .. math::  \widetilde G = -\frac{1}{2}\big(G + G^\dagger\big) 
# 
#   gives a Hermitian matrix with eigenvalues :math:`x_k`. One can perform an analogous construction
#   with an additional :math:`S` gate in :math:`W` to obtain the imaginary part :math:`y_k`.
# 
# Theorem 4 of [#nlat]_ shows that :math:`\widetilde G` and :math:`\widetilde G'` are
# :math:`(1,1,0)`-block encodings of these diagonal matrices. Remarkably, they can be implemented with
# just four applications of :math:`U` and :math:`U^\dagger` and :math:`\mathcal{O}(n)` additional one‑
# and two‑qubit gates. We will now implement this block‑encoding for a small example.
# 


# Necessary imports for the NTCA demo.
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from numpy.polynomial import Chebyshev, Polynomial
import matplotlib.pyplot as plt

# Display version numbers
print("PennyLane version:", qml.__version__)



# -----------------------------------------------------------------------------
#  Implementation of the block‑encoding for real or imaginary
#  parts of amplitudes.

# Controlled-Z on multiple controls.  control_values specify which bit value
# selects the gate; default is all zeros.
def MultiControlledZ(wires, control_values=None):
    if control_values is None:
        control_values = [0] * (len(wires) - 1)
    qml.ctrl(qml.Z(wires=wires[-1]), control=wires[:-1], control_values=control_values)

# R_gate implements the reflection R used in the block construction.
def R_gate(wires):
    n = len(wires)//2
    # wires[0] holds the selector qubit used for the reflection
    qml.PauliX(wires=wires[0])
    MultiControlledZ(wires=wires[n+1:] + [wires[0]])
    qml.PauliX(wires=wires[0])

# Apply U on the data register conditioned on ancilla B=0.  U can be a callable
# or an Operator.  Additional arguments are passed through via *args, **kwargs.
def Uc_on_data(base, wires, *args, **kwargs):
    n = len(wires)//2
    data, B = wires[n+1:], wires[n]
    if isinstance(base, qml.typing.TensorLike):
        qml.ControlledQubitUnitary(base, control_wires=B, wires=data, control_values=[0], unitary_check=True)
    elif isinstance(base, qml.operation.Operator) or callable(base):
        qml.ctrl(base, control=B, control_values=[0])(wires=data, *args, **kwargs)

# Adjoint of U on the data register controlled on ancilla B=0.
def Uc_adj_on_data(base, wires, *args, **kwargs):
    n = len(wires)//2
    data, B = wires[n+1:], wires[n]
    if isinstance(base, qml.typing.TensorLike):
        qml.adjoint(qml.ControlledQubitUnitary)(base, control_wires=B, wires=data, control_values=[0], unitary_check=True)
    elif isinstance(base, qml.operation.Operator) or callable(base):
        qml.ctrl(qml.adjoint(base), control=B, control_values=[0])(wires=data, *args, **kwargs)

# Copy the ancilla B qubit into the address register (controlled Toffoli chain).
# This coherently adds or subtracts the basis state |k> to the prepared state.
def C_to_data(wires):
    n = len(wires)//2
    for i in range(n):
        qml.Toffoli(wires=[wires[n], wires[i], wires[n+i+1]])

# The adjoint of C_to_data, reversing the coherent copy.
def C_adj_to_data(wires):
    n = len(wires)//2
    for i in range(n-1, -1, -1):
        qml.Toffoli(wires=[wires[n], wires[i], wires[n+i+1]])

# One step of the W operator.  If p_flag=1 an S gate is applied to the ancilla B
# to pick up a phase for the imaginary part. 
def W_block(base, wires, p_flag=0, *args, **kwargs):
    n = len(wires)//2
    B = wires[n]
    qml.Hadamard(wires=B)
    Uc_on_data(base, wires, *args, **kwargs)
    C_to_data(wires)
    if bool(p_flag):
        qml.S(wires=B)
    qml.Hadamard(wires=B)

# Adjoint of W_block.
def W_block_adj(base, wires, p_flag=0, *args, **kwargs):
    n = len(wires)//2
    B = wires[n]
    qml.Hadamard(wires=B)
    if bool(p_flag):
        qml.adjoint(qml.S)(wires=B)
    C_adj_to_data(wires)
    Uc_adj_on_data(base, wires, *args, **kwargs)
    qml.Hadamard(wires=B)

# G_block implements the operator G = W S0 W^† Z_B.  Its adjoint is defined
# similarly.  See Eq. (9) of [#nlat]_.
def G_block(base, wires, p_flag=0, *args, **kwargs):
    n = len(wires)//2
    qml.PauliZ(wires=wires[n])
    W_block_adj(base, wires, p_flag, *args, **kwargs)
    R_gate(wires)
    W_block(base, wires, p_flag, *args, **kwargs)

# Adjoint of G_block.
def G_block_adj(base, wires, p_flag=0, *args, **kwargs):
    W_block_adj(base, wires, p_flag, *args, **kwargs)
    R_gate(wires)
    W_block(base, wires, p_flag, *args, **kwargs)
    qml.PauliZ(wires=wires[len(wires)//2])

# AmplitudeBlockEncoding wraps the above primitives to prepare the block
# encoding G tilde or G' tilde. The ancilla_wires comprise the selector,
# address register and B; data wires contain the data qubits.
# p_flag=1 switches to the imaginary part.
def AmplitudeBlockEncoding(U_callable, wires, ancilla_wires, p_flag=0, *args, **kwargs):
    assert len(ancilla_wires) == len(wires) + 2, "ancillas must be selector + addr + B"
    # Hadamard on selector
    qml.Hadamard(wires=ancilla_wires[0])
    # bundle = [addr..., B] + data
    bundle = ancilla_wires[1:] + list(wires)
    # W operation
    W_block(U_callable, bundle, p_flag, *args, **kwargs)
    # Controlled G and G† depending on selector
    qml.ctrl(G_block,     control=ancilla_wires[0], control_values=[0])(U_callable, bundle, p_flag, *args, **kwargs)
    qml.ctrl(G_block_adj, control=ancilla_wires[0], control_values=[1])(U_callable, bundle, p_flag, *args, **kwargs)
    # Hadamard on selector
    qml.Hadamard(wires=ancilla_wires[0])
    # Undo W
    W_block_adj(U_callable, bundle, p_flag, *args, **kwargs)
    # Clean up selector by conjugating with XZX
    qml.PauliX(wires=ancilla_wires[0]); qml.PauliZ(wires=ancilla_wires[0]); qml.PauliX(wires=ancilla_wires[0])


######################################################################
# The above code implements the block‑encoding :math:`\widetilde G` for the real part of the
# amplitudes. For demonstrations, we want a spectrum of :math:`\mathrm{Re}(c_k)` values so the
# polynomial’s effect is visible across the different basis states :math:`k`. A short, normalized
# vector with both real and imaginary entries, such as :math:`[0.4, 0.3, 0.2, 0.1, 0.15i, 0, 0, 0]`,
# is ideal. It creates a long-tail distribution while keeping :math:`|\mathrm{Re}(c_k)| \le 0.4`,
# which is safely inside the :math:`[-1,1]` range required for QSVT. We also need to specify the
# arrangement of qubits: for :math:`n` data qubits the ancilla register consists of a selector qubit,
# an :math:`n`-qubit address register, and a single ancilla :math:`B`. The total number of qubits is
# thus :math:`2n+2`.
# 
# Below we create a simple block‑encoding for :math:`n=3` and inspect its matrix to confirm that its
# eigenvalues correspond to the data amplitudes.
# 

#------------- device -------------
n_data = 3
ancilla_wires = list(range(0, n_data + 2))   # selector, addr, B
data_wires    = list(range(n_data + 2, 2*n_data + 2))
all_wires     = ancilla_wires + data_wires
num_total_qubits = len(all_wires)
dev = qml.device("default.qubit", wires=num_total_qubits)

# Sample amplitude vector
psi_data = np.array([0.4, 0.3, 0.2, 0.1, 0.15j, 0, 0, 0], dtype=complex)
psi_data = psi_data / np.linalg.norm(psi_data)

print("Input state psi_data (c_k):")
print(psi_data)
print("-" * 20)

# State preparation function
def U_data_prep(wires, state_vec=None):
    if state_vec is None:
        # Default state prep if none provided (less useful for verification)
        dim = 2**len(wires)
        v = pnp.ones(dim) # Changed to ones for simplicity if needed
        v = v/pnp.linalg.norm(v)
        qml.StatePrep(v, wires=wires)
    else:
        qml.StatePrep(pnp.array(state_vec, dtype=complex), wires=wires)


def get_block_encoding_operator(state_vec_in,
                                data_wires=data_wires,
                                ancilla_wires=ancilla_wires,
                                p_flag=0):
    # The AmplitudeBlockEncoding operator.
    # U_data_prep is the 'base' callable passed to AmplitudeBlockEncoding.
    # wires are data_wires, ancilla_wires are ancilla_wires.
    return qml.prod(AmplitudeBlockEncoding)(U_data_prep,
                           wires=data_wires,
                           ancilla_wires=ancilla_wires,
                           p_flag=p_flag, # p_flag=0 for real part
                           state_vec=state_vec_in)
# --- Verification ---

# 1. Get the matrix of the full unitary G_tilde
# We pass psi_data so U_data_prep inside AmplitudeBlockEncoding uses it
full_block_matrix = qml.matrix(get_block_encoding_operator(psi_data))
print(f"Shape of full block-encoding unitary (G_tilde): {full_block_matrix.shape}")
print("-" * 20)

# 2. Extract the encoded operator H = <0|G_tilde|0>
# The encoded operator H acts on the subspace [addr..., B, data...]
# It's the top-left block of the full matrix when selector=0
U_full = full_block_matrix.reshape([2**(n_data+2), 2**n_data, 2**(n_data+2), 2**n_data])
H_encoded_matrix = U_full[0, :, 0, :]  # <0|U|0> block
print(f"Shape of extracted encoded operator (H): {H_encoded_matrix.shape}")
print("-" * 20)

# 3. Calculate eigenvalues of H
# Use eigvalsh since we expect it to be Hermitian
try:
    eigenvalues_H = np.linalg.eigvalsh(H_encoded_matrix)
except np.linalg.LinAlgError:
    print("Warning: eigvalsh failed, matrix might not be perfectly Hermitian numerically.")
    print("Trying eigvals instead...")
    eigenvalues_H = np.linalg.eigvals(H_encoded_matrix)

# 4. Compare eigenvalues
# Round eigenvalues due to numerical precision issues
rounded_eigenvalues = np.round(eigenvalues_H, decimals=8)

# Find the unique eigenvalues found in the simulation
unique_simulated_eigenvalues = np.unique(rounded_eigenvalues)

print(f"Unique eigenvalues found in H:")
print(unique_simulated_eigenvalues)
print("-" * 20)

@qml.qnode(dev)
def get_block_encoding_qnode(state_vec_in):
    AmplitudeBlockEncoding(U_data_prep,
                           wires=data_wires,
                           ancilla_wires=ancilla_wires,
                           p_flag=0, # p_flag=0 for real part
                           state_vec=state_vec_in)
    return qml.state()

qml.draw_mpl(get_block_encoding_qnode)(psi_data)

######################################################################
# Resource analysis
# ~~~~~~~~~~~~~~~~~
# 
# ``qml.specs(get_block_encoding_qnode)(psi_data)`` prints a **static resource summary** for the
# Amplitude Block Encoding call used in the real-only experiment.
# 
# - ``num_wires`` / ``num_device_wires`` / ``num_tape_wires``: how many qubits are in play (ancillas +
#   DATA).
# - ``depth``: circuit depth at the current abstraction level.
# - ``shots``: ``None`` means **statevector** simulation; if you switch to a finite-shots device,
#   specs will reflect sampling.
# - ``diff_method`` / ``gradient_fn``: how gradients would be computed if you trained parameters.
# 
# **Reading specs**: for algorithmic understanding, the most meaningful items are the number of
# **ancillas** (``selector`` + ``address`` + ``signal`` :math:`B`), and whether you are in
# **statevector** vs **shots** mode.
# 

qml.specs(get_block_encoding_qnode)(psi_data)

######################################################################
# 1.2. From block‑encoding to nonlinear functions via QSVT
# --------------------------------------------------------
# 
# Once the real and imaginary parts of the amplitudes are available as the eigenvalues of Hermitian
# block‑encoded matrices, one can apply **Quantum Singular Value Transformation (QSVT)** to implement
# polynomial functions of those eigenvalues. QSVT is a generalisation of Quantum Signal Processing and
# can apply any bounded polynomial :math:`P(x)` to the singular values of a block‑encoded matrix by
# composing controlled reflections and single‑qubit phase rotations. For more details about
# implementing polynomials of block-encoded Hamiltonians, block-encoding operators, and rotation
# operators, see the :doc:`demos/tutorial_intro_qsvt`.
# Theorem 5 of [#nlat]_ states that if :math:`P(x)` and :math:`Q(x)` can be
# approximated by degree‑\ :math:`d` polynomials :math:`P'` and :math:`Q'` to precision
# :math:`\epsilon/(4N)`, then the NTCA task can be realised with
# 
# - :math:`\mathcal{O}\big(d\,\gamma\,\sqrt{N} / \;\Vert P'(x_k)+Q'(y_k)\Vert_2\big)` applications of
#   :math:`U` and :math:`U^\dagger`, and
# - :math:`\mathcal{O}\big(nd\,\gamma\,\sqrt{N} / \;\Vert P'(x_k)+Q'(y_k)\Vert_2\big)` one‑ and
#   two‑qubit gates,
# 
# where :math:`\gamma=\max_{x\in[-1,1]}\{|P(x)|,|Q(x)|\}`. Intuitively, the degree of the polynomial
# controls the accuracy of the nonlinearity and the cost of the QSVT sequence.
# 
# In PennyLane, QSVT is implemented via ``qml.QSVT``. Given a block‑encoding and a list of phase
# angles returned by ``qml.poly_to_angles``, the call
# 
# .. code:: python
# 
#    qml.QSVT(block_encoding, projectors)
# 
# applies a sequence of reflections and controlled‐phase rotations on the signal qubit that effects
# the polynomial transformation. The phase angles are computed from the Chebyshev expansion of the
# target function.
# 

######################################################################
# We want a smooth, odd, saturating activation. We choose the *Chebyshev approximation* of
# :math:`\tanh(\alpha x)` on :math:`[-1, 1]`: 1. Fit :math:`\tanh(\alpha x)` with a Chebyshev series
# of **odd degree** :math:`d`. 2. Convert the Chebyshev series to the ordinary power basis to get
# coefficients :math:`P(x) = \sum_{k=0}^{d} P_k x^k`, then **zero out even coefficients** to enforce
# oddness. 3. **Scale** the coefficients so that :math:`\max_{x \in [-1, 1]} |P(x)| \le \frac{1}{4}`.
# 

def build_poly(kind="tanh_cheb", deg=7, alpha=1.5, raw_coeffs=None, gridN=2001):
    assert (deg % 2 == 1) or (kind in ("linear","raw")), "Use odd degree for odd activations"
    if kind == "raw":
        coeffs = np.array(raw_coeffs, dtype=float)
    elif kind == "linear":
        coeffs = np.zeros(max(2, deg+1)); coeffs[1] = 1.0
    else:
        xs = np.linspace(-1, 1, gridN); ys = np.tanh(alpha*xs)
        cheb = Chebyshev.fit(xs, ys, deg); poly = cheb.convert(kind=Polynomial)
        coeffs = np.array(poly.coef)
        for k in range(0, len(coeffs), 2): coeffs[k] = 0.0
    xs = np.linspace(-1, 1, 4001); vals = np.polyval(coeffs[::-1], xs)
    gamma = max(1e-12, np.max(np.abs(vals)))
    return coeffs/(4*gamma)

def target_real(psi, P):
    x = np.real(psi)
    Pval = np.zeros_like(x, dtype=float)
    for k, ck in enumerate(P): Pval += ck * (x**k)
    Pval = np.clip(Pval, -0.25, 0.25)
    w = (np.abs(psi)**2) * (np.abs(Pval)**2)
    Z = w.sum(); return w/(Z if Z>0 else 1.0)

def target_complex(psi, P, Q):
    a = np.real(psi); b = np.imag(psi)
    Pval = np.zeros_like(a, dtype=float); Qval = np.zeros_like(b, dtype=float)
    for k, ck in enumerate(P): Pval += ck * (a**k)
    for k, ck in enumerate(Q): Qval += ck * (b**k)
    Pval = np.clip(Pval, -0.25, 0.25); Qval = np.clip(Qval, -0.25, 0.25)
    f = Pval + 1j*Qval
    w = (np.abs(psi)**2) * (np.abs(f)**2)
    Z = w.sum(); return w/(Z if Z>0 else 1.0)

def tvd(a,b): return 0.5*np.sum(np.abs(a-b))

def bits_of(i, m): return [(i>>b)&1 for b in range(m)][::-1]

def data_distribution(psi, ancilla_wires, n_data, require_B_zero=False, require_selector_zero=True):
    num_anc = len(ancilla_wires)
    probs = np.zeros(2**n_data); succ=0.0
    for s in range(2**(num_anc + n_data)):
        bits = bits_of(s, num_anc + n_data)
        sel = bits[0]; Bbit = bits[1+n_data]
        if (not require_selector_zero or sel==0) and (not require_B_zero or Bbit==0):
            k=0
            for b in bits[num_anc:]: k=(k<<1)|b
            p = float(np.abs(psi[s])**2)
            probs[k]+=p; succ+=p
    if succ>0: probs/=succ
    return probs, succ

def postselected_amps(psi, ancilla_wires, n_data, require_B_zero=True, require_selector_zero=True):
    num_anc = len(ancilla_wires)
    amps = np.zeros(2**n_data, dtype=complex); succ=0.0
    for s in range(2**(num_anc + n_data)):
        bits = bits_of(s, num_anc + n_data)
        sel = bits[0]; Bbit = bits[1+n_data]
        if (not require_selector_zero or sel==0) and (not require_B_zero or Bbit==0):
            k=0
            for b in bits[num_anc:]: k=(k<<1)|b
            a = psi[s]; amps[k]+=a; succ+=float(np.abs(a)**2)
    return amps, succ

# ------------- device -------------
n_data = 3
ancilla_wires = list(range(0, n_data + 2))   # selector, addr, B
data_wires    = list(range(n_data + 2, 2*n_data + 2))
all_wires     = ancilla_wires + data_wires
num_total_qubits = len(all_wires)
dev = qml.device("default.qubit", wires=num_total_qubits)

# Sample amplitude vector
psi_data = np.array([0.4, 0.3, 0.2, 0.1, 0.15j, 0, 0, 0], dtype=complex)
psi_data = psi_data / np.linalg.norm(psi_data)

# ------------- build block-encoding operator -------------
block_real = get_block_encoding_operator(psi_data, p_flag=0)
block_imag = get_block_encoding_operator(psi_data, p_flag=1)

# ------------- build polynomial and phase sequence -------------
P = build_poly(kind="tanh_cheb", deg=7, alpha=1.5)
angles_P = qml.poly_to_angles(P, "QSVT", angle_solver="root-finding")

def make_proj(angles, wires):
    dim = 2 ** len(wires)
    return [qml.PCPhase(phi, dim=dim, wires=wires) for phi in angles]

projs = make_proj(angles_P, all_wires)

# ------------- define QSVT circuit -------------
@qml.qnode(dev)
def qsvt_real(projectors):
    # prepare nontrivial input
    U_data_prep(wires=data_wires, state_vec=psi_data)
    qml.QSVT(block_real, projectors)
    return qml.state()

psi_qsvt = qsvt_real(projs)

# === Step 5: measure postselected branch ===
obs, succ = data_distribution(
    psi_qsvt, ancilla_wires, n_data,
    require_B_zero=True, require_selector_zero=True
)

# ------------- classical target and comparison -------------
tgt = target_real(psi_data, P)
tvd_val = tvd(obs, tgt)

plt.figure()
x = np.arange(len(obs))
plt.plot(x, tgt, 'o-', label='Target (real)')
plt.plot(x, obs, 'x-', label=f'Observed (real)')
plt.xlabel("Index k (DATA)")
plt.ylabel("Probability")
plt.title(f"Real-only QSVT tanh-approx | TVD={tvd_val:.4f}")
plt.legend()
plt.show()


######################################################################
# 2. Application: a two-block quantum classifier with nonlinear activations
# -------------------------------------------------------------------------
# 
# | To showcase **NTCA as a genuine nonlinear activation layer** inside a *trainable* quantum model,
# | we now construct a small quantum neural network composed of **two stacked NTCA blocks** —
# | the quantum analogue of two :math:`\tanh` neurons in a classical MLP.
# 
# Instead of a fixed amplitude encoding, each block begins with a **parameterized embedding unitary**
# 
# .. math::
# 
# 
#    U_{\text{embed}}(x; \theta_0, \phi_0)
# 
# | that learns how to map the classical input feature (x) onto a quantum state.
# | Each NTCA block then applies its nonlinear transformation through block-encoding and QSVT,
#   producing a learned “activation” in amplitude space.
# 
# The model contains **seven trainable parameters** in total:
# 
# - **:math:`\theta_{0,1}, \phi_{0,1}` and :math:`\theta_{0,2}, \phi_{0,2}`** — parameters of the two
#   embedding unitaries that learn distinct feature projections of the same input (x).
# - **:math:`\theta_{\text{addr,1}}` and :math:`\theta_{\text{addr,2}}`** — pre-mix rotations on the
#   address ancilla for each NTCA block, determining how the encoded amplitude is routed through the
#   nonlinear layer.
# - **:math:`\beta_{\text{post}}`** — a final post-rotation on the data qubit, serving as a learnable
#   bias term in the readout stage.
# 
# Together, these form a *quantum two-layer perceptron*:
# 
# .. math::
# 
# 
#    x 
#    \;\longrightarrow\;
#    U_{\text{embed}}^{(1)}(x)
#    \;\xrightarrow{\text{NTCA}}\;
#    U_{\text{embed}}^{(2)}(x)
#    \;\xrightarrow{\text{NTCA}}\;
#    \text{measurement}.
# 
# We train this model on a **one-dimensional binary classification dataset** where the positive class
# occupies two **disjoint regions** on the real line — an arrangement that cannot be represented by
# any single-layer (monotonic) model. The two NTCA activations allow the circuit to construct **two
# separate decision lobes** in amplitude space, illustrating how quantum nonlinearities can emulate
# the expressive power of multi-neuron classical networks while remaining fully coherent.
# 

import pennylane as qml
import pennylane.numpy as pnp
import matplotlib.pyplot as plt
from tqdm import trange

# ---------- Device ----------
nq = 1
anc_q = list(range(0, nq + 2))      # [sel, addr, B]
data_q = list(range(nq + 2, 2*nq + 2))
dev_q  = qml.device("default.qubit", wires=len(anc_q)+len(data_q))

# ---------- Parameterized data embedding ----------
def U_from_x_kw(*, wires, x, theta0, phi0):
    qml.RY(theta0 * x, wires=wires)
    qml.RZ(phi0 * x, wires=wires)

# ---------- NTCA polynomial activation (same as before) ----------
P = build_poly(deg=7, alpha=1.5)
angles = qml.poly_to_angles(P, "QSVT", angle_solver="root-finding")
B_wire = anc_q[-1]
projs_q = [qml.PCPhase(phi, dim=2, wires=[B_wire]) for phi in angles]

# ---------- One NTCA block builder ----------
def build_block_for_x(x, theta0, phi0):
    return qml.prod(AmplitudeBlockEncoding)(
        U_from_x_kw, wires=data_q, ancilla_wires=anc_q,
        p_flag=0, x=x, theta0=theta0, phi0=phi0
    )

# ---------- Two-layer QNN ----------
@qml.qnode(dev_q, interface="autograd")
def qnn_probs_2tanh(x, theta_addr1, theta_addr2, beta_post,
                    theta0_1, phi0_1, theta0_2, phi0_2):
    # ---- First "tanh neuron" ----
    block1 = build_block_for_x(x, theta0_1, phi0_1)
    qml.RY(theta_addr1, wires=anc_q[1])
    qml.QSVT(block1, projs_q)

    # ---- Second "tanh neuron" ----
    block2 = build_block_for_x(x, theta0_2, phi0_2)
    qml.RY(theta_addr2, wires=anc_q[1])
    qml.QSVT(block2, projs_q)

    # ---- Readout ----
    qml.RY(beta_post, wires=data_q[0])
    return qml.probs(wires=range(len(anc_q)+len(data_q)))

# ---------- Predict probability of y=1 ----------
def predict_p1(x, params):
    theta_addr1, theta_addr2, beta_post, theta0_1, phi0_1, theta0_2, phi0_2 = params
    p = qnn_probs_2tanh(x, theta_addr1, theta_addr2, beta_post,
                        theta0_1, phi0_1, theta0_2, phi0_2)
    n_tot = len(anc_q)+len(data_q)
    p1, succ = 0.0, 0.0
    for s in range(2**n_tot):
        bits = [(s>>b)&1 for b in range(n_tot)][::-1]
        sel = bits[0]; data_bit = bits[len(anc_q)]
        Bbit = bits[1+nq]
        if sel == 0 and Bbit == 0:
            succ += p[s]
            if data_bit == 1:
                p1 += p[s]
    return p1/(succ + 1e-12)


# ---------- Dataset ----------
X = pnp.linspace(-1.0, 1.0, 20)
y = pnp.array([(1 if xi > 0.4 or xi < -0.4 else 0) for xi in X])

# ---------- Initialize parameters ----------
params = [
    pnp.array(0.2, requires_grad=True),  # theta_addr1
    pnp.array(-0.3, requires_grad=True), # theta_addr2
    pnp.array(0.0, requires_grad=True),  # beta_post
    pnp.array(1.0, requires_grad=True),  # theta0_1
    pnp.array(0.5, requires_grad=True),  # phi0_1
    pnp.array(-1.2, requires_grad=True), # theta0_2
    pnp.array(0.8, requires_grad=True)   # phi0_2
]

# ---------- Loss ----------
def loss(*params):
    eps = 1e-8
    pis = pnp.stack([predict_p1(float(xi), params) for xi in X])
    return -pnp.mean(y*pnp.log(pis+eps) + (1-y)*pnp.log(1-pis+eps))

# ---------- Train ----------
grad = qml.grad(loss)
lr = 0.1; hist=[]
for _ in trange(80, desc="Training 2-layer NTCA-QNN", ncols=70):
    g = grad(*params)
    for i in range(len(params)):
        params[i] -= lr * g[i]
    hist.append(loss(*params))

p_after = [float(predict_p1(float(xi), params)) for xi in X]


plt.figure(); plt.plot(hist)
plt.xlabel("iteration"); plt.ylabel("BCE loss"); plt.title("Training curve (2-NTCA QNN)"); plt.show()

plt.figure()
plt.plot(X, p_after, "s-",  label="QNN output")
plt.plot(X, y, "k:", label="target")
plt.xlabel("x"); plt.ylabel("p(y=1|x)")
plt.title("Decision boundary learned by 2-NTCA QNN")
plt.legend(); plt.ylim(-0.05,1.05); plt.show()

# ---------- Confusion matrix ----------
preds = [1 if p>=0.5 else 0 for p in p_after]
TP = sum(int(pi==1 and yi==1) for pi,yi in zip(preds,y))
TN = sum(int(pi==0 and yi==0) for pi,yi in zip(preds,y))
FP = sum(int(pi==1 and yi==0) for pi,yi in zip(preds,y))
FN = sum(int(pi==0 and yi==1) for pi,yi in zip(preds,y))
print("Confusion matrix (rows=true, cols=pred):")
print(f"[[TN={TN}, FP={FP}],[FN={FN}, TP={TP}]]   Accuracy={(TP+TN)/len(y):.3f}")


######################################################################
# 3. Conclusion
# -------------
# 
# In this demo we prepared implementation of **nonlinear transformation of complex amplitudes** (NTCA)
# using the block‑encoding described by [#nlat]_ and the Quantum Singular Value
# Transformation. We then constructed a small concrete example. We encoded the real part of quantum
# state amplitudes into the spectrum of a Hermitian operator using four calls to a state preparation
# oracle and :math:`\mathcal{O}(n)` additional gates. Applying QSVT with phase angles derived from a
# Chebyshev approximation of the hyperbolic tangent implemented an odd, bounded polynomial activation
# function. Finally, we demonstrated how to integrate this nonlinear activation into a simple quantum
# classifier trained on a handful of points.
# 
# NTCA provides a route to implement nonlinear functions within the otherwise linear framework of
# quantum mechanics, at the cost of post‑selection and increased circuit depth. While our example used
# a single data qubit and an odd activation, the approach scales to larger registers and general
# complex functions :math:`P(x)+Q(y)`. Combining NTCA with parameterised circuits opens promising
# possibilities for quantum machine learning.
# 
# References
# ----------
#
# .. [#nlat]
#
#     Naixu Guo, Kosuke Mitarai, Keisuke Fujii,
#     "Nonlinear transformation of complex amplitudes via quantum singular value transformation".
#     `Physical Review Research <https://link.aps.org/doi/10.1103/PhysRevResearch.6.043227>`__, 2024
#