export FastStandardSyndrome, AbstractFastSyndrome

abstract type AbstractFastSyndrome <: SyndromeCircuit end
struct FastStandardSyndrome <: AbstractFastSyndrome end

function apply_fast_error!(noise_model::Nothing,
                           state::ChpState, zx_counts_out::Vector{Int},
                           kind::Symbol, qubits)
    # Apply no error when no noise model
end
function apply_fast_error!(noise_model::NoiseModel,
                           state::ChpState, zx_counts_out::Vector{Int},
                           kind::Symbol, qubits)
    p = noise_model.errors[kind]
    # TODO: rng
    z_count = x_count = 0
    for q in qubits
        r = rand(rng, Float64)
        r >= p && return
        # Apply X, Y, or Z with (1/3)p probability
        if r < (2/3)*p
            state.x[q] ⊻= true  # Z gate
            z_count += 1
        end
        if r >= (1/3)*p
            state.z[q] ⊻= true  # X gate
            x_count += 1
        end
    end
    zx_counts_out[2] += z_count  # Z errors cause X syndromes
    zx_counts_out[1] += x_count
    nothing
end

function exec_syndrome_layer(noise_model::Union{NoiseModel, Nothing},
                             syndrome_circuit::AbstractFastSyndrome,
                             run::CodeDistanceRun,
                             layer_i::Int)
    ctx = run.ctx
    state = run.state
    # Apply errors
    for q in ctx.data_qubits
        apply_fast_error!(noise_model, state, run.zx_error_counts,
                         :uniform_data, (q,))
    end
    # Run circuit
    for info in ctx.z_plaq_info
        anc = info.ancilla
        for dat in info.data
            state.z[anc] ⊻= state.z[dat]  # CNOT
        end
    end
    for info in ctx.x_plaq_info
        anc = info.ancilla
        for dat in info.data
            state.x[anc] ⊻= state.x[dat]  # CNOT
        end
    end
    # Apply errors
    for q in ctx.anc_qubits
        apply_fast_error!(noise_model, state, run.zx_meas_error_counts,
                         :uniform_anc, (q,))
    end
    # Measure
    for (i, info) in enumerate(ctx.z_plaq_info)
        meas = state.z[info.ancilla]  # Measure
        state.z[info.ancilla] = false  # Reset
        run.z_syndromes[i, layer_i] = run.z_prev[i] ⊻ meas
        run.z_prev[i] = meas
    end
    for (i, info) in enumerate(ctx.x_plaq_info)
        meas = state.x[info.ancilla]  # Measure
        state.x[info.ancilla] = false  # Reset
        run.x_syndromes[i, layer_i] = run.x_prev[i] ⊻ meas
        run.x_prev[i] = meas
    end
    nothing
end

function simulate_syndrome_run(syndrome_circuit::AbstractFastSyndrome,
                               run::CodeDistanceRun)
    ctx = run.ctx
    run.state.z[1:run.ctx.num_qubits] .= false  # Clean start
    run.state.x[1:run.ctx.num_qubits] .= false
    run.zx_error_counts .= 0
    run.zx_meas_error_counts .= 0
    exec_syndrome_layer(nothing, syndrome_circuit, run, 1)
    for i in 1:ctx.m_dist+1
        # Noise-free end layer
        noise = i == ctx.m_dist+1 ? nothing : ctx.noise_model
        exec_syndrome_layer(noise, syndrome_circuit, run, i)
    end
    run.z_syndromes, run.x_syndromes
end
