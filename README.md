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

  
          ![equation](https://latex.codecogs.com/svg.image?|x\rangle=\frac{1}{\|x\|}\sum_{i=1}^{d}x_i|i\rangle&space;)


where:
- ||x|| is the Euclidean norm of x
- |i> represents the i-th basis state
- This ensures the state is normalized: <x|x> = 1


In practice, we simulate this on a classical computer by normalizing and padding/truncating vectors to match 2ⁿ dimensions.

---

## Quantum Kernel

Once data is amplitude-encoded, we can define a quantum-inspired kernel based on simulating state overlap: K(x, x') = |⟨x|x'⟩|² this mesures similarity between 2 states.

Even though this is computed classically, it retains the properties of a quantum kernel and can be used directly with SVM.

---

## Why This Kernel is Useful

- Maps data into a high-dimensional Hilbert space, making linearly inseparable classes separable.
- Provides a novel similarity measure based on overlaps instead of Euclidean distance.
- Even simulated classically, it emulates a quantum-inspired feature map, making it a valid SVM kernel.

---

## Benchmarks

We tested Quantum Kernel SVM against **RBF SVM** and **Random Forest**.

**Key Observations:**

- Works well on medium-high dimensional datasets with enough samples per class (Digits, Olivetti).  
- Struggles on low-dimensional or noisy datasets (Iris, LFW).  
- Execution time is competitive on small datasets; Random Forest is slower for large datasets.

## Benchmark Results

| Dataset        | Quantum-kernel SVM        | RBF SVM baseline          | Random Forest             |
|----------------|--------------------------|--------------------------|---------------------------|
| Iris           | 0.6800 ± 0.1204, 0.00s   | 0.9600 ± 0.0389, 0.00s  | 0.9533 ± 0.0340, 0.16s   |
| Wine           | 0.9046 ± 0.0222, 0.00s   | 0.9833 ± 0.0222, 0.00s  | 0.9887 ± 0.0138, 0.20s   |
| Breast Cancer  | 0.7996 ± 0.0179, 0.01s   | 0.9772 ± 0.0163, 0.01s  | 0.9526 ± 0.0180, 0.32s   |
| Digits         | 0.9783 ± 0.0087, 0.13s   | 0.9839 ± 0.0060, 0.13s  | 0.9750 ± 0.0047, 0.57s   |
| LFW            | 0.5939 ± 0.0202, 0.25s   | 0.7523 ± 0.0199, 1.81s  | 0.6266 ± 0.0138, 5.06s   |
| Olivetti       | 0.9300 ± 0.0257, 0.03s   | 0.9425 ± 0.0359, 0.46s  | 0.9500 ± 0.0112, 3.64s   |



### Key Takeaways
- Quantum-inspired kernels can emulate high-dimensional mappings that are potentially useful for SVM classification.  
- Classical simulations of quantum kernels are fast for moderate dataset sizes, but real quantum advantage would require actual quantum hardware.  
- Performance heavily depends on dataset size, feature dimensionality, and class balance.




