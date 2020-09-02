export FastStandardSyndrome, AbstractFastSyndrome

abstract type AbstractFastSyndrome <: SyndromeCircuit end
struct FastStandardSyndrome <: AbstractFastSyndrome end

function apply_fast_error!(noise_params::Nothing,
                           state::ChpState, zx_counts_out::Vector{Int},
                           kind::Symbol, qubits)
    # Apply no error when no noise model
end
function apply_fast_error!(noise_params::Dict{Symbol, Float64},
                           state::ChpState, zx_counts_out::Vector{Int},
                           kind::Symbol, qubits)
    p = noise_params[kind]
    p == 0 && return
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

function apply_fast_cnot_error!(noise_params::Nothing,
                                state::ChpState,
                                zx_counts_out1::Vector{Int},
                                zx_counts_out2::Vector{Int},
                                kind::Symbol, q1::Int, q2::Int)
    # Apply no error when no noise model
end
function apply_fast_cnot_error!(noise_params::Dict{Symbol, Float64},
                                state::ChpState,
                                zx_counts_out1::Vector{Int},
                                zx_counts_out2::Vector{Int},
                                kind::Symbol, q1::Int, q2::Int)
    p = noise_params[kind]
    p == 0 && return
    # TODO: rng
    z_count1 = x_count1 = 0
    z_count2 = x_count2 = 0
    r = rand(rng, Float64)
    r >= p && return
    # Apply IX, IY, IZ, XI, XX, XY, ..., ZY, ZZ with (1/15)p probability
    c = rand(0:14)
    c1 = div(c, 4)
    c2 = mod(c, 4)
    if c1 < 2
        state.x[q1] ⊻= true  # Z gate
        z_count1 += 1
    end
    if 1 <= c1 < 3
        state.z[q1] ⊻= true  # X gate
        x_count1 += 1
    end
    if c2 < 2
        state.x[q2] ⊻= true  # Z gate
        z_count2 += 1
    end
    if 1 <= c2 < 3
        state.z[q2] ⊻= true  # X gate
        x_count2 += 1
    end
    zx_counts_out1[2] += z_count1  # Z errors cause X syndromes
    zx_counts_out1[1] += x_count1
    zx_counts_out2[2] += z_count2  # Z errors cause X syndromes
    zx_counts_out2[1] += x_count2
    nothing
end

function exec_syndrome_layer(
        noise_params::Union{Nothing, Dict{Symbol, Float64}},
        syndrome_circuit::AbstractFastSyndrome,
        run::CodeDistanceRun,
        layer_i::Int)
    ctx = run.ctx
    state = run.state
    # Apply errors
    p_data = layer_i == 1 ? :p_data_layer1 : :p_data
    for q in ctx.data_qubits
        apply_fast_error!(noise_params, state, run.zx_error_counts,
                          p_data, (q,))
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
    for info in ctx.z_plaq_info
        anc = info.ancilla
        for (i, dat) in enumerate(info.data)
            apply_fast_cnot_error!(noise_params, state, run.zx_error_counts,
                                   run.zx_meas_error_counts,
                                   (i==1 ? :p_cnot1 : :p_cnot), dat, anc)
        end
    end
    for info in ctx.x_plaq_info
        anc = info.ancilla
        for (i, dat) in enumerate(info.data)
            apply_fast_cnot_error!(noise_params, state, run.zx_meas_error_counts,
                                   run.zx_error_counts,
                                   (i==1 ? :p_cnot1 : :p_cnot), anc, dat)
        end
    end
    for info in ctx.z_plaq_info
        apply_fast_error!(noise_params, state, run.zx_meas_error_counts,
                          :p_anc_z, (info.ancilla,))
    end
    for info in ctx.x_plaq_info
        apply_fast_error!(noise_params, state, run.zx_meas_error_counts,
                          :p_anc_x, (info.ancilla,))
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
        noise = i == ctx.m_dist+1 ? nothing : run.sim_noise_params
        exec_syndrome_layer(noise, syndrome_circuit, run, i)
    end
    run.z_syndromes, run.x_syndromes
end
