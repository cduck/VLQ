# Virtualized Logical Qubits: A 2.5D Architecture for Error-Corrected Quantum Computing

This is the source code repository for the paper
[Virtualized Logical Qubits: A 2.5D Architecture for Error-Corrected Quantum Computing][arxiv]
published in the proceedings of MICRO '20, the 53rd IEEE/ACM International Symposium on Microarchitecture, October 2020.

Follow the instructions below to reproduce the simulation results from the paper.  See the [documentation](#documentation) for implementation details and guidance on extending it for new fault-tolerant architectures.

[arxiv]: https://arxiv.org/...


## Install

1. Clone this repository
    ```bash
    git clone https://github.com/cduck/vlq
    cd vlq
    ```

2. Install Julia (tested with 1.4.2): [julialang.org/downloads](https://julialang.org/downloads/)

3. Install required Julia packages (run from the `vlq/` directory)
    ```bash
    julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'
    ```
    or from the julia REPL
    ```
    ] activate .; instantiate
    ```


## Usage

Run the following Julia code in a REPL (start one with `julia --project=.`)

```julia
# Setup
include("src/make_plots.jl"); using .MakePlots
MakePlots.setup(num_workers=16)  # Setup 16 worker processes

# Run the simulations
job_id = "all_plots"
samples = 1000*2000  # How many samples, ~1.15 CPU-hours per 100000 samples
dists = [3:2:11...]  # Which code distances to simulate
plots = [1:12...]  # Which of the 12 plots to generate
MakePlots.dist_calc_all(job_id, dists, samples, 1:12)  # Start computing

# Plot the results as they are computed
# Run plot_finished() any time to check the progress of a plot
for plot_i in 1:12
    MakePlots.plot_finished(job_id, plot_i)
end
```


## Documentation

See [the paper][arxiv] for an overview of quantum error correction and the surface code.
The following describes the components of simulation and error correction and how to extend them to evaluate new architectures.

### Simulation

`src/evaluate_qec.jl`, `src/fast_syndrome.jl`

The simulation component substitutes for running the error syndrome circuits on real hardware.
Because H and CNOT gates and the Pauli errors X, Y, and Z are all Clifford gates, we use a [CHP simulator](https://github.com/cduck/ChpSim.jl) instead of a general quantum circuit simulator (implemented in `src/evaluate_qec.jl:exec_syndrome_layer`).  Certain properties of the surface code allow us to speed up simulation even more by separately tracking X and Z flips caused by CNOTs and errors and get the same outcome as the Clifford simulator (implemented in `src/fast_syndrome.jl:exec_syndrome_layer()`).

To measure the logical error rate for a logical qubit with a given distance, d, we initialize d^2 data qubits and d^2-1 ancilla qubits, run one cycle of syndrome extraction noise-free, d cycles of noisy syndrome extraction, then a final noise-free cycle.
Each syndrome extraction cycle consists of ancilla preparation, a CNOT from each data to all its neighboring ancilla, then ancilla measurement.  Errors can occur at any time and we count how many X and Z errors occur on data qubits.

The measurement results for all the plaquettes (one per ancilla) for every time step are stored in an array.  When no errors are present, the measurement results will stay constant over time.  An error will cause one or more to flip so we take the XOR over the time axis to isolate error syndromes.  We keep X and Z syndromes in two separate arrays as we will correct X and Z separately.  These two arrays of Bools are then given to the error matching step.

### Error Matching

`src/evaluate_qec.jl`

The job of error matching is to determine what errors occurred on which data or ancilla based only on the X or Z syndrome array generated earlier.  One way to do this is with [minimum weight perfect matching](https://en.wikipedia.org/wiki/Matching_(graph_theory)) (although real time applications use faster but less accurate algorithms).  We use the [BlossomV](http://pub.ist.ac.at/~vnk/papers/blossom5-TR-Sep9.pdf) [library](https://github.com/mlewe/BlossomV.jl) for this and only need to compute the edge weights.

To do this we construct a graph where each node is a Z plaquette at a time step, t, plus some boundary nodes.  Edges are between neighboring plaquettes in both space and time.  Boundary nodes connect to Z plaquettes that have data qubits not shared with any other Z plaquette.  Edge weights are the negative log probability of an error on the shared data qubit (for space edges) or the negative log probability of an error on the ancilla or measurement error (for time edges).

This graph is not the input to BlossomV; shortest path lengths through it are used as the edge weights in a new complete graph.  This new complete graph has one node for each True value in the syndrome array (corresponding to a plaquette that detected an error).  Odd graphs have no perfect matching so we conditionally add a dummy boundary node to ensure an even number.

BlossomV takes this complete graph and tells us the most likely error chains through the 3D volume of the surface code we simulated.  The exact paths don't matter, only the parity of the length.  Two Z errors are parity 0, three are parity 1, four are 0, etc.  To determine if a Z error occurred we compare the number of errors injected with the number of errors detected (modulo 2).

The entire matching process happens once each for X and Z.  By simulating and matching thousands of times, we estimate the logical error rate of the given configuration and noise model.

### Device Characteristics and Noise Model

`src/noise_model.jl`

To perform the simulation we need to know how often to randomly apply noise and for matching, we need to calculate accurate edge weights.
Both of these are informed by the noise model and other device parameters such as gate duration.
For Virtualized Logical Qubits, code distance also affects noise because the refresh time depended on the code distance.

Custom `SyndromeCircuit` subtypes are the easiest way to specify new types of devices and architectures, requiring only three methods to fully describe the device.
There are five already defined for Virtualized Logical Qubits: `TypicalSyndrome` is the 2D surface code and the others are for variations of the 2.5D architecture described in the paper.
The simulator uses the method `simulation_noise_parameters()` to determine what noise to inject.  Error matching uses `matching_space_edge()` and `matching_time_edge()` to assign edge weights.  See `src/noise_model.jl` for examples.  These methods should agree closely, otherwise matching will perform poorly.

### Error Threshold and Sensitivity Results

`src/make_plots.jl`

To make sense of the error rates we calculate, we generate error threshold and sensitivity plots.  Threshold plots have one curve per code distance where the x-axis is a simplification of the device's physical error rate and the y-axis is the estimated logical error rate.  The crossing point of all the curves is the threshold.  Sensitivity plots are similar but the x-axis sweeps over a single device parameter or error rate.

Computing the data points for these plots can be very time consuming, even with very fast simulation and matching.  `src/make_plots.jl` defines the 12 plots in the paper but can be modified to generate other sets of plots.  It uses Julia's distributed computing to parallelize the work.
