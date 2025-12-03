"""
NISQ Error Metrics Module for FedMO-DRLQ
=========================================
Implements error-aware state representation incorporating NISQ device characteristics:
- T1/T2 coherence times
- Gate errors (single-qubit and two-qubit)
- Readout errors
- Calibration drift indicators

This addresses Gap 1: No DRL Approach Incorporates NISQ Error Characteristics

Author: Sandhya (NIT Sikkim)
Reference: FedMO-DRLQ Research Gap Analysis
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import time


class DeviceType(Enum):
    """Types of quantum devices"""
    SUPERCONDUCTING = "superconducting"
    TRAPPED_ION = "trapped_ion"
    PHOTONIC = "photonic"


@dataclass
class NISQDeviceMetrics:
    """
    NISQ Device Error Metrics Container.
    
    Based on real IBM Quantum device characteristics:
    - washington (127 qubits)
    - kolkata (27 qubits) 
    - hanoi (27 qubits)
    - perth (7 qubits)
    - lagos (7 qubits)
    
    State representation includes:
    s_error = [ε²q_median, T₂_median, ε_r_median, t_calibration, ε̇₂q, QV, CLOPS]
    """
    
    device_name: str
    device_type: DeviceType = DeviceType.SUPERCONDUCTING
    num_qubits: int = 27
    
    # Two-qubit gate error (median across all CX gates)
    two_qubit_error_median: float = 0.01  # 1% typical
    two_qubit_error_std: float = 0.005
    
    # Single-qubit gate error (median)
    single_qubit_error_median: float = 0.0003  # 0.03% typical
    single_qubit_error_std: float = 0.0001
    
    # Coherence times (microseconds)
    t1_median: float = 150.0  # 150 μs typical
    t1_std: float = 50.0
    t2_median: float = 100.0  # 100 μs typical  
    t2_std: float = 30.0
    
    # Readout error (median)
    readout_error_median: float = 0.015  # 1.5% typical
    readout_error_std: float = 0.005
    
    # Calibration information
    last_calibration_time: float = 0.0  # Unix timestamp
    calibration_interval: float = 3600.0  # seconds (1 hour)
    
    # Error drift rate (change per second)
    error_drift_rate: float = 0.0001
    
    # Benchmark metrics
    quantum_volume: int = 128  # Quantum Volume
    clops: float = 2000.0  # Circuit Layer Operations Per Second
    
    def get_time_since_calibration(self) -> float:
        """Get time since last calibration in seconds"""
        return time.time() - self.last_calibration_time
    
    def get_calibration_factor(self) -> float:
        """
        Get calibration quality factor (0-1).
        Decreases as time since calibration increases.
        """
        t_since_cal = self.get_time_since_calibration()
        # Exponential decay with half-life of calibration_interval
        decay_rate = np.log(2) / self.calibration_interval
        return np.exp(-decay_rate * t_since_cal)
    
    def get_current_two_qubit_error(self) -> float:
        """
        Get current estimated two-qubit error accounting for drift.
        """
        t_since_cal = self.get_time_since_calibration()
        drift = self.error_drift_rate * t_since_cal
        return min(self.two_qubit_error_median + drift, 0.1)  # Cap at 10%
    
    def to_state_vector(self) -> np.ndarray:
        """
        Convert to error-aware state vector.
        
        State vector: [ε²q_median, T₂_median, ε_r_median, t_calibration, ε̇₂q, QV, CLOPS]
        
        All values are normalized to [0, 1] range for neural network input.
        """
        # Normalized values
        epsilon_2q_norm = min(self.get_current_two_qubit_error() / 0.05, 1.0)  # Normalize to 5% max
        t2_norm = min(self.t2_median / 300.0, 1.0)  # Normalize to 300 μs max
        readout_norm = min(self.readout_error_median / 0.1, 1.0)  # Normalize to 10% max
        calibration_norm = self.get_calibration_factor()  # Already 0-1
        drift_norm = (self.error_drift_rate + 0.001) / 0.002  # Normalize [-0.001, 0.001] to [0, 1]
        qv_norm = min(np.log2(self.quantum_volume + 1) / 10, 1.0)  # Log scale, normalize to QV=1024
        clops_norm = min(self.clops / 5000.0, 1.0)  # Normalize to 5000 CLOPS max
        
        return np.array([
            epsilon_2q_norm,
            t2_norm,
            readout_norm,
            calibration_norm,
            drift_norm,
            qv_norm,
            clops_norm
        ], dtype=np.float32)


# Pre-defined IBM Quantum device configurations
IBM_DEVICE_CONFIGS = {
    "washington": NISQDeviceMetrics(
        device_name="ibm_washington",
        num_qubits=127,
        two_qubit_error_median=0.0095,
        t1_median=180.0,
        t2_median=120.0,
        readout_error_median=0.012,
        quantum_volume=64,
        clops=1800.0
    ),
    "kolkata": NISQDeviceMetrics(
        device_name="ibm_kolkata",
        num_qubits=27,
        two_qubit_error_median=0.008,
        t1_median=200.0,
        t2_median=150.0,
        readout_error_median=0.010,
        quantum_volume=128,
        clops=2200.0
    ),
    "hanoi": NISQDeviceMetrics(
        device_name="ibm_hanoi",
        num_qubits=27,
        two_qubit_error_median=0.009,
        t1_median=170.0,
        t2_median=110.0,
        readout_error_median=0.014,
        quantum_volume=64,
        clops=2000.0
    ),
    "perth": NISQDeviceMetrics(
        device_name="ibm_perth",
        num_qubits=7,
        two_qubit_error_median=0.006,
        t1_median=250.0,
        t2_median=180.0,
        readout_error_median=0.008,
        quantum_volume=32,
        clops=2500.0
    ),
    "lagos": NISQDeviceMetrics(
        device_name="ibm_lagos",
        num_qubits=7,
        two_qubit_error_median=0.007,
        t1_median=220.0,
        t2_median=160.0,
        readout_error_median=0.009,
        quantum_volume=32,
        clops=2400.0
    )
}


class NISQErrorSimulator:
    """
    Simulator for NISQ device error dynamics.
    
    Simulates time-varying error characteristics including:
    - Random fluctuations in error rates
    - Calibration drift
    - T1/T2 decoherence variations
    """
    
    def __init__(self, device_name: str, seed: Optional[int] = None):
        """
        Initialize error simulator.
        
        Args:
            device_name: Name of IBM device to simulate
            seed: Random seed for reproducibility
        """
        if device_name not in IBM_DEVICE_CONFIGS:
            raise ValueError(f"Unknown device: {device_name}. Available: {list(IBM_DEVICE_CONFIGS.keys())}")
        
        self.base_metrics = IBM_DEVICE_CONFIGS[device_name]
        self.rng = np.random.default_rng(seed)
        self.current_metrics = self._create_current_metrics()
        self.last_update_time = time.time()
        
    def _create_current_metrics(self) -> NISQDeviceMetrics:
        """Create a copy of base metrics with random variations"""
        return NISQDeviceMetrics(
            device_name=self.base_metrics.device_name,
            device_type=self.base_metrics.device_type,
            num_qubits=self.base_metrics.num_qubits,
            two_qubit_error_median=self._add_noise(
                self.base_metrics.two_qubit_error_median,
                self.base_metrics.two_qubit_error_std
            ),
            single_qubit_error_median=self._add_noise(
                self.base_metrics.single_qubit_error_median,
                self.base_metrics.single_qubit_error_std
            ),
            t1_median=self._add_noise(
                self.base_metrics.t1_median,
                self.base_metrics.t1_std
            ),
            t2_median=self._add_noise(
                self.base_metrics.t2_median,
                self.base_metrics.t2_std
            ),
            readout_error_median=self._add_noise(
                self.base_metrics.readout_error_median,
                self.base_metrics.readout_error_std
            ),
            last_calibration_time=time.time(),
            calibration_interval=self.base_metrics.calibration_interval,
            error_drift_rate=self.rng.uniform(-0.0005, 0.0005),
            quantum_volume=self.base_metrics.quantum_volume,
            clops=self._add_noise(self.base_metrics.clops, 200.0)
        )
    
    def _add_noise(self, mean: float, std: float) -> float:
        """Add Gaussian noise to a value, ensuring non-negative result"""
        return max(0, self.rng.normal(mean, std * 0.2))  # Use 20% of std for small variations
    
    def update(self, dt: float = 1.0):
        """
        Update device metrics based on time passage.
        
        Args:
            dt: Time step in seconds
        """
        # Apply random walk to error rates
        self.current_metrics.two_qubit_error_median += self.rng.normal(0, 0.0001) * dt
        self.current_metrics.two_qubit_error_median = np.clip(
            self.current_metrics.two_qubit_error_median, 0.001, 0.05
        )
        
        # Coherence times can vary slightly
        self.current_metrics.t2_median += self.rng.normal(0, 1.0) * dt
        self.current_metrics.t2_median = np.clip(
            self.current_metrics.t2_median, 20.0, 300.0
        )
        
        self.last_update_time = time.time()
    
    def calibrate(self):
        """Simulate recalibration - reset to near-optimal values"""
        self.current_metrics = self._create_current_metrics()
        self.current_metrics.last_calibration_time = time.time()
    
    def get_state_vector(self) -> np.ndarray:
        """Get current error-aware state vector"""
        return self.current_metrics.to_state_vector()
    
    def get_metrics(self) -> NISQDeviceMetrics:
        """Get current metrics object"""
        return self.current_metrics


class FidelityEstimator:
    """
    Estimates circuit fidelity based on NISQ error characteristics.
    
    Fidelity estimation model:
    F = F_gate * F_readout * F_decoherence
    
    where:
    - F_gate = (1 - ε₁q)^n₁ * (1 - ε₂q)^n₂
    - F_readout = (1 - ε_r)^n_measure  
    - F_decoherence = exp(-t_circuit / T₂)
    """
    
    @staticmethod
    def estimate_fidelity(
        metrics: NISQDeviceMetrics,
        n_single_qubit_gates: int,
        n_two_qubit_gates: int,
        n_measurements: int,
        circuit_depth: int,
        gate_time_ns: float = 500.0  # Typical 2-qubit gate time in nanoseconds
    ) -> float:
        """
        Estimate expected circuit fidelity.
        
        Args:
            metrics: NISQ device metrics
            n_single_qubit_gates: Number of single-qubit gates
            n_two_qubit_gates: Number of two-qubit gates (CX, CZ, etc.)
            n_measurements: Number of measurement operations
            circuit_depth: Circuit depth
            gate_time_ns: Two-qubit gate duration in nanoseconds
            
        Returns:
            Estimated fidelity in [0, 1]
        """
        # Gate fidelity
        f_single = (1 - metrics.single_qubit_error_median) ** n_single_qubit_gates
        f_two = (1 - metrics.get_current_two_qubit_error()) ** n_two_qubit_gates
        f_gate = f_single * f_two
        
        # Readout fidelity
        f_readout = (1 - metrics.readout_error_median) ** n_measurements
        
        # Decoherence fidelity (simplified model)
        circuit_time_us = (circuit_depth * gate_time_ns) / 1000.0  # Convert to microseconds
        t2_us = metrics.t2_median
        f_decoherence = np.exp(-circuit_time_us / t2_us) if t2_us > 0 else 0.0
        
        # Combined fidelity
        fidelity = f_gate * f_readout * f_decoherence
        
        return max(0.0, min(1.0, fidelity))  # Clip to [0, 1]
    
    @staticmethod
    def estimate_from_circuit_info(
        metrics: NISQDeviceMetrics,
        circuit_gates: Dict[str, int],
        circuit_depth: int
    ) -> float:
        """
        Estimate fidelity from circuit gate count dictionary.
        
        Args:
            metrics: NISQ device metrics
            circuit_gates: Dictionary of gate counts (e.g., {'cx': 50, 'rz': 100, ...})
            circuit_depth: Circuit depth
            
        Returns:
            Estimated fidelity
        """
        # Count two-qubit gates (CX, CZ, CNOT, etc.)
        two_qubit_keys = ['cx', 'cnot', 'cz', 'swap', 'iswap', 'ecr', 'rzz', 'rxx', 'ryy']
        n_two_qubit = sum(circuit_gates.get(k, 0) for k in two_qubit_keys)
        
        # Count single-qubit gates
        single_qubit_keys = ['x', 'y', 'z', 'h', 'rx', 'ry', 'rz', 's', 't', 'sx', 'sdg', 'tdg', 'u', 'u1', 'u2', 'u3', 'p']
        n_single_qubit = sum(circuit_gates.get(k, 0) for k in single_qubit_keys)
        
        # Count measurements
        n_measurements = circuit_gates.get('measure', 1)
        
        return FidelityEstimator.estimate_fidelity(
            metrics=metrics,
            n_single_qubit_gates=n_single_qubit,
            n_two_qubit_gates=n_two_qubit,
            n_measurements=n_measurements,
            circuit_depth=circuit_depth
        )


class EnergyEstimator:
    """
    Estimates energy consumption for quantum circuit execution.
    
    Energy model considers:
    - Cryogenic cooling power
    - Gate operation energy
    - Classical control electronics
    """
    
    # Base power consumption for different device components (Watts)
    CRYOGENIC_POWER = 15000.0  # 15 kW for dilution refrigerator
    CONTROL_POWER_PER_QUBIT = 50.0  # 50 W per qubit for control electronics
    GATE_ENERGY_NJ = 10.0  # nanojoules per gate operation
    
    @staticmethod
    def estimate_energy(
        metrics: NISQDeviceMetrics,
        n_total_gates: int,
        execution_time_seconds: float,
        num_shots: int = 1024
    ) -> float:
        """
        Estimate total energy consumption in Joules.
        
        Args:
            metrics: NISQ device metrics
            n_total_gates: Total number of gates
            execution_time_seconds: Total execution time
            num_shots: Number of circuit repetitions
            
        Returns:
            Energy consumption in Joules
        """
        # Cryogenic power (continuous during execution)
        e_cryogenic = EnergyEstimator.CRYOGENIC_POWER * execution_time_seconds
        
        # Control electronics power
        e_control = EnergyEstimator.CONTROL_POWER_PER_QUBIT * metrics.num_qubits * execution_time_seconds
        
        # Gate operation energy
        e_gates = EnergyEstimator.GATE_ENERGY_NJ * n_total_gates * num_shots * 1e-9  # Convert nJ to J
        
        return e_cryogenic + e_control + e_gates
    
    @staticmethod
    def normalize_energy(energy_joules: float, max_energy: float = 100000.0) -> float:
        """Normalize energy to [0, 1] range"""
        return min(energy_joules / max_energy, 1.0)


# Convenience functions
def get_device_metrics(device_name: str) -> NISQDeviceMetrics:
    """Get pre-configured metrics for an IBM device"""
    if device_name not in IBM_DEVICE_CONFIGS:
        raise ValueError(f"Unknown device: {device_name}")
    return IBM_DEVICE_CONFIGS[device_name]


def get_error_aware_state_dim() -> int:
    """Get the dimension of the error-aware state vector"""
    return 7  # [ε²q, T₂, ε_r, t_cal, ε̇₂q, QV, CLOPS]