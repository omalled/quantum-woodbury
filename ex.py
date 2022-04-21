import time
import json
from matplotlib import pyplot
from math import ceil
from collections import defaultdict
from qiskit.compiler import transpile, assemble
from qiskit.providers.ibmq.managed import IBMQJobManager
import numpy
from qiskit.providers.aer import AerSimulator, StatevectorSimulator
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library.standard_gates import HGate, XGate, YGate, ZGate
import qiskit.quantum_info as qi
import qiskit
from qiskit import IBMQ

#constructs a circuit that can estimate <psi|u|psi> where |psi>=psi_prep|0> using the Hadamard test
#it can give either the real or imaginary part based on whether complex_test is False or True, respectively
def hadamard_test_circuit(u, psi_prep, complex_test=False):
    u_controlled = u.control(1)
    ht_circuit = QuantumCircuit(u_controlled.num_qubits, 1)
    ht_circuit.h(0)
    if complex_test:
        ht_circuit.p(-numpy.pi / 2, 0)
    ht_circuit.append(psi_prep, list(range(1, u_controlled.num_qubits)))
    ht_circuit.append(u_controlled, list(range(u_controlled.num_qubits)))
    ht_circuit.h(0)
    ht_circuit.measure(0, 0)
    return ht_circuit

def my_getcounts(counts, i):
    if i in counts:
        return counts[i]
    else:
        return 0

def hadamard_test_expval(counts, bias_matrix):
    c0 = my_getcounts(counts, '0')
    c1 = my_getcounts(counts, '1')
    c0_corrected, c1_corrected = numpy.linalg.solve(bias_matrix, numpy.transpose(numpy.array([c0, c1])))
    return ((c0_corrected - c1_corrected) / (c0 + c1))

#estimates <psi|u|psi> using an exact statevector simulator
#this gives the correct answer to more-or-less machine precision
def expval(u, psi_prep):
    svsim = StatevectorSimulator()
    psi_sv = qiskit.execute(psi_prep, svsim).result().get_statevector()
    u_psi_sv = qiskit.execute(psi_prep.compose(u), svsim).result().get_statevector()
    return numpy.vdot(psi_sv.data, u_psi_sv.data)

#estimates <psi|u|psi> using the hadamard test for both the real and imaginary parts
def hadamard_expval_circuits(u, psi_prep):
    return [hadamard_test_circuit(u, psi_prep), hadamard_test_circuit(u, psi_prep, complex_test=True)]

#computes <y|x> using the hadamard test
def hadamard_inner_products(y_preps, x_preps, shots, backend, bias_matrices=defaultdict(lambda: numpy.matrix([[1.0, 0], [0, 1]])), zne_repeats=0, isreal=False):
    hadamard_circs = []
    for i in range(len(y_preps)):
        qc = QuantumCircuit(x_preps[i].num_qubits)
        qc.append(x_preps[i], list(range(x_preps[i].num_qubits)))
        qc.append(y_preps[i].inverse(), list(range(x_preps[i].num_qubits)))
        if isreal:
            hadamard_circs.append(hadamard_expval_circuits(qc.to_gate(), y_preps[i])[0])
        else:
            hadamard_circs.extend(hadamard_expval_circuits(qc.to_gate(), y_preps[i]))
    counts, transpiled_circs = run_circuits(hadamard_circs, backend, shots, zne_repeats=zne_repeats)
    retval = []
    for i in range(len(y_preps)):
        q1 = transpiled_circs[-2][-1][1][0].index
        q2 = transpiled_circs[-1][-1][1][0].index
        if isreal:
            retval.append(hadamard_test_expval(counts[i], bias_matrices[q1]))
        else:
            retval.append(complex(hadamard_test_expval(counts[2 * i + 0], bias_matrices[q1]),
                                  hadamard_test_expval(counts[2 * i + 1], bias_matrices[q2])))
    return retval

def measurement_bias_matrices(backend, shots):
    if shots > 0:
        bias_circuits = []
        for qubit in range(backend.configuration().n_qubits):
            qc = QuantumCircuit(backend.configuration().n_qubits, 1)
            qc.measure(qubit, 0)
            bias_circuits.append(qc)
            qc = QuantumCircuit(backend.configuration().n_qubits, 1)
            qc.x(qubit)
            qc.measure(qubit, 0)
            bias_circuits.append(qc)
        counts, _ = run_circuits(bias_circuits, backend, shots, transpile_circs=False)
        bias_matrices = {}
        for i in range(backend.configuration().n_qubits):
            A11 = my_getcounts(counts[2 * i], '0') / shots
            A21 = my_getcounts(counts[2 * i], '1') / shots
            A12 = my_getcounts(counts[2 * i + 1], '0') / shots
            A22 = my_getcounts(counts[2 * i + 1], '1') / shots
            bias_matrices[i] = numpy.matrix([[A11, A12], [A21, A22]])
        return bias_matrices
    else:
        bias_matrices = {}
        properties = backend.properties()
        for i in range(backend.configuration().n_qubits):
            p01 = properties.qubit_property(i)['prob_meas0_prep1'][0]
            p10 = properties.qubit_property(i)['prob_meas1_prep0'][0]
            A11 = 1.0 - p10
            A21 = p10
            A12 = p01
            A22 = 1.0 - p01
            bias_matrices[i] = numpy.matrix([[A11, A12], [A21, A22]])
        return bias_matrices

        return

def blowchunks(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def repeat_zne(circ, repeats):
    new_circ = QuantumCircuit(circ.num_qubits, circ.num_clbits)
    for inst, qargs, cargs in circ.data[0:-1]:#leave the last instruction off, which should be the measurement instruction
        if inst.name != "sx":#skip the repeats for sx, since the inverse isn't in the basis gates
            for i in range(repeats):
                new_circ._append(inst, qargs, cargs)
                new_circ._append(inst.inverse(), qargs, cargs)
        new_circ._append(inst, qargs, cargs)
    inst, qargs, cargs = circ.data[-1]#now add the measurement back on
    new_circ._append(inst, qargs, cargs)
    return new_circ

def run_circuits(circs, backend, shots, transpile_circs=True, zne_repeats=0):
    if transpile_circs:
        circs_transpiled = transpile(circs, backend=backend, optimization_level=3, seed_transpiler=0)#setting the seed is important for the ZNE
    else:
        circs_transpiled = circs
    circs_transpiled = list(map(lambda x: repeat_zne(x, zne_repeats), circs_transpiled))
    #circs_transpiled = transpile(circs, backend=backend, optimization_level=0, seed_transpiler=0)#now do the transpilation to get it back to the native gate set, but set the optimization level to 0
    conf = backend.configuration()
    experiments_per_circ = ceil(shots / conf.max_shots)
    repeated_circs = [circ for circ in circs_transpiled for i in range(experiments_per_circ)]
    if hasattr(conf, 'max_experiments'):
        max_experiments = conf.max_experiments
    else:
        max_experiments = len(repeated_circs)
    job_chunks = blowchunks(repeated_circs, max_experiments)
    all_jobsets = []
    for job_chunk in job_chunks:#send all the circuits off to run
        all_jobsets.append(backend.run(job_chunk, shots=ceil(shots / experiments_per_circ)))
    all_counts = []
    for i in range(len(all_jobsets)):#get all the counts
        result = all_jobsets[i].result()
        for j in range(len(job_chunks[i])):
            all_counts.append(result.get_counts(j))
    collated_counts = []
    for i in range(len(circs)):
        collated_counts.append({'0': 0, '1': 0})
        for j in range(experiments_per_circ):
            these_counts = all_counts[j + i * experiments_per_circ]
            collated_counts[-1]['0'] += my_getcounts(these_counts, '0')
            collated_counts[-1]['1'] += my_getcounts(these_counts, '1')
    return collated_counts, circs_transpiled

def rank1_batch(z_prep, b_prep, v_prep, u_prep, alpha, beta, shots, backend, **kwargs):
    u1s = [z_prep, v_prep, v_prep, z_prep]
    u2s = [b_prep, b_prep, u_prep, u_prep]
    zb, vb, vu, zu = hadamard_inner_products(u1s, u2s, shots, backend, **kwargs)
    return zb - alpha * beta * vb / (1 + alpha * beta * vu) * zu

def hadamards(n):
    c = QuantumCircuit(n)
    for i in range(n):
        c.h(i)
    return c

zx_exact = 1 / 2

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-lanl', group='lanl', project='quantum-optimiza')
#backend = provider.get_backend('ibmq_bogota')
#bogota = provider.get_backend('ibmq_bogota')
hardware = provider.get_backend('ibm_auckland')
#hardware = provider.get_backend('ibmq_montreal')


#backend = qiskit.providers.aer.QasmSimulator.from_backend(hardware)
backend = hardware
#bias_matrices = measurement_bias_matrices(backend, 10 ** 5)
bias_matrices = measurement_bias_matrices(backend, 0)
num_samples = 10 ** 5
results_exact_sim = []
results_hardware_sim = []
results_hardware_sim_me = []
results_hardware_sim_me_zne = []
alpha = 1
beta = 1
#ns = list(range(1, 27))
#ns = [2, 4, 8, 16, 26]
#ns = [26, 16, 12, 8, 4, 2]
#ns = [12, 8, 4, 2]
ns = [16, 20, 24, 26]
#ns = [26]
for n in ns:
    print(n)
    t0 = time.time()
    z_prep = hadamards(n).to_gate()
    b_prep = hadamards(n).to_gate()
    u_prep = hadamards(n).to_gate()
    v_prep = hadamards(n).to_gate()
    zx_h = rank1_batch(z_prep, b_prep, v_prep, u_prep, alpha, beta, num_samples, backend, isreal=True)
    results_hardware_sim.append(zx_h)
    print(f"{zx_h}: relative error with {num_samples} samples: {numpy.abs(zx_h - zx_exact) / numpy.abs(zx_exact)} ({backend.name()})")
    zx_h1 = rank1_batch(z_prep, b_prep, v_prep, u_prep, alpha, beta, num_samples, backend, bias_matrices=bias_matrices, isreal=True)
    results_hardware_sim_me.append(zx_h1)
    print(f"{zx_h1}: relative error with {num_samples} samples: {numpy.abs(zx_h1 - zx_exact) / numpy.abs(zx_exact)} ({backend.name()} + measurement error correction)")
    zx_h3 = rank1_batch(z_prep, b_prep, v_prep, u_prep, alpha, beta, num_samples, backend, bias_matrices=bias_matrices, zne_repeats=1, isreal=True)
    print(f"{zx_h3}: relative error with {num_samples} samples: {numpy.abs(zx_h3 - zx_exact) / numpy.abs(zx_exact)} ({backend.name()} + measurement error correction + 1 ZNE repeat)")
    zx_zne = zx_h1 + (zx_h1 - zx_h3) / (1 - 3) * (0 - 1)
    results_hardware_sim_me_zne.append(zx_zne)
    print(f"{zx_zne}: relative error with {num_samples} samples: {numpy.abs(zx_zne - zx_exact) / numpy.abs(zx_exact)} (ZNE extrapolation)")
    t1 = time.time()
    print(t1 - t0, "seconds")

d = {'ns': ns, 'results_hardware_sim': results_hardware_sim, 'results_hardware_sim_me': results_hardware_sim_me, 'results_hardware_sim_me_zne': results_hardware_sim_me_zne}
with open(f"results_hardware_{ns}.json", 'w') as outfile:
    json.dump(d, outfile)
