# Quantum-inspired Kernel SVM Benchmarking on scikit-learn Datasets

## Project Overview

This project investigates the performance of a **quantum-inspired kernel SVM** compared to classical machine learning models (RBF SVM and Random Forest) on standard scikit-learn datasets such as Iris, Wine, Breast Cancer, Digits, Olivetti Faces, and LFW.  

We explore how **amplitude encoding**, inspired by quantum computing, can map classical feature vectors into a **higher-dimensional Hilbert space** to compute a **quantum kernel** and use it for classification. Execution times of all models are also compared.

---

The idea comes from the growing interest in quantum machine learning (QML), where quantum computers or quantum-inspired algorithms can potentially offer advantages for classification and pattern recognition.  

While real quantum hardware is limited, quantum-inspired algorithms like amplitude encoding and quantum kernels allow us to simulate quantum computations classically. This enables experimentation and benchmarking without needing a quantum computer.
---

## Quantum Fundamentals (Simulated)

Quantum computing relies on the qubit, which can exist in a superposition of 0 and 1:  |ψ⟩ = α|0⟩ + β|1⟩

where α, β ∈ ℂ and |α|² + |β|² = 1.  

For a system of `n` qubits, the state vector lives in a 2ⁿ-dimensional Hilbert space.

**Amplitude encoding:** maps classical feature vectors x ∈ ℝᵈ into a quantum state:  
$$
|x\rangle = \frac{1}{\|x\|} \sum_{i=1}^{d} x_i \, |i\rangle
$$




