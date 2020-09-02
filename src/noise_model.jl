export TypicalSyndrome, NaturalAllAtOnce, NaturalInterleaved, CompactAllAtOnce,
    CompactInterleaved,
    make_noise_model_for_paper


# The different circuits evaluated
struct TypicalSyndrome <: AbstractFastSyndrome end
struct NaturalAllAtOnce <: AbstractFastSyndrome end
struct NaturalInterleaved <: AbstractFastSyndrome end
struct CompactAllAtOnce <: AbstractFastSyndrome end
struct CompactInterleaved <: AbstractFastSyndrome end


### Make the noise model used in the paper
p0 = 0.1
starting_model = Dict{Symbol, Float64}(
    :t1_t => 100_000,  # ns
    :t1_c => 1_000_000,  # ns
    :dur_t => 50,  # ns
    :dur_tt => 200,  # ns
    :dur_tc => 200,  # ns
    :dur_loadstore => 150,  # ns
    :dur_meas => 200,  # ns
    :p_t => p0 / 10,
    :p_tt => p0,
    :p_tc => p0,
    :p_loadstore => p0,
    :p_meas => p0,
    :cavity_depth => 10,
)
const_keys = [:dur_tt, :dur_t, :dur_tc, :dur_loadstore, :dur_meas,
              :cavity_depth]
coherence_keys = [:t1_t, :t1_c]
error_rate_keys = [:p_t, :p_tt, :p_tc, :p_loadstore, :p_meas]
sensitivity_base_p = 2e-3

function make_noise_model_for_paper(base_error::Float64, override_pairs=())
    error_factor = base_error / p0
    actual_model = Dict{Symbol, Float64}()
    for k in const_keys
        actual_model[k] = starting_model[k]
    end
    for k in coherence_keys
        actual_model[k] = starting_model[k] / error_factor
    end
    for k in error_rate_keys
        actual_model[k] = starting_model[k] * error_factor
    end
    for (k, v) in override_pairs
        actual_model[k] = v
    end
    NoiseModel(actual_model)
end


### Util functions
"""
    combine_flip_probs(p1, p2, ...)

Return the probability that an odd number of events occur
with the given probabilities.
"""
function combine_flip_probs(p_flip::Float64...)
    p_total = 0
    for p in p_flip
        p_total = p_total + p - 2*p_total*p
    end
    p_total
end

"""
    combine_error_probs(p1, p2, ...)

Return the probability that at least one event occurs
with the given probabilities.
"""
function combine_error_probs(p_err::Float64...)
    1 - prod(1 .- p_err)
end

function coherence_error(t1, duration)
    1 - MathConstants.e ^ (-duration / t1)
end


### Calculate edge weights and simulation noise parameters

function calculate_qubit_error_single_pauli(model::NoiseModel;
        t_t::Float64 = 0.0,
        t_c::Float64 = 0.0,
        n_t::Int = 0,
        n_tt::Int = 0,
        n_tc::Int = 0,
        n_loadstore::Int = 0,
        n_meas::Int = 0,
    )
    combine_flip_probs(
        2/3 * coherence_error(model.t1_t, t_t),
        2/3 * coherence_error(model.t1_c, t_c),
        repeat([2/3*model.p_t], n_t)...,
        repeat([8/15*model.p_tt], n_tt)...,
        repeat([8/15*model.p_tc], n_tc)...,
        repeat([2/3*model.p_loadstore], n_loadstore)...,
        repeat([model.p_meas], n_meas)...,
    )
end

# Typical Syndrome
# Z: One CNOT per data, then measure
# X: Hadamard, one CNOT (reverse dir) per data, hadamard, measure
function matching_space_edge(::TypicalSyndrome, model::NoiseModel,
                             is_x::Bool, dist::Int, first_layer::Bool)
    # Mainly 4 CNOTs; one per plaquette
    p = calculate_qubit_error_single_pauli(model,
        t_t = 4*model.dur_tt + 2*model.dur_t + model.dur_meas,
        n_t = 0,
        n_tt = 4,
    )
    -log(p)
end
function matching_time_edge(::TypicalSyndrome, model::NoiseModel,
                            is_x::Bool, dist::Int, first_layer::Bool)
    # Mainly 4 CNOTs
    # Also two Hadamards if is X
    p = calculate_qubit_error_single_pauli(model,
        t_t = 4*model.dur_tt + (is_x ? 2*model.dur_t : 0),
        n_t = (is_x ? 2 : 0),
        n_tt = 4,
        n_meas = 1,
    )
    -log(p)
end
function simulation_noise_parameters(::TypicalSyndrome, model::NoiseModel,
                                     ctx::CodeDistanceSim)
    t_data = 4*model.dur_tt + 2*model.dur_t + model.dur_meas
    t_anc_z = 4*model.dur_tt
    t_anc_x = 4*model.dur_tt + 2*model.dur_t
    Dict{Symbol, Float64}(
        :p_data_layer1 => coherence_error(model.t1_t, t_data),
        :p_data => coherence_error(model.t1_t, t_data),
        :p_anc_z => combine_error_probs(
            calculate_qubit_error_single_pauli(model,
                t_t = t_anc_z,
                n_t = 0),
            model.p_meas),
        :p_anc_x => combine_error_probs(
            calculate_qubit_error_single_pauli(model,
                t_t = t_anc_x,
                n_t = 2),
            model.p_meas),
        :p_cnot1 => model.p_tt,
        :p_cnot => model.p_tt,
    )
end

# NaturalAllAtOnce
# Z: One CNOT per data, then measure
# X: Hadamard, one CNOT (reverse dir) per data, hadamard, measure
# Extra store-load error in first layer and stored for (cavity-1)*d layers
function matching_space_edge(::NaturalAllAtOnce, model::NoiseModel,
                             is_x::Bool, m_dist::Int, first_layer::Bool)
    # Estimate of error on data qubit
    t_round = 4*model.dur_tt + 2*model.dur_t + model.dur_meas
    p = if first_layer
        calculate_qubit_error_single_pauli(model,
            t_t = t_round,
            t_c = ((model.cavity_depth-1)
                   * (m_dist * t_round + 2*model.dur_loadstore)),
            n_t = 0,
            n_tt = 4,
            n_loadstore = 2,
        )
    else
        calculate_qubit_error_single_pauli(model,
            t_t = t_round,
            t_c = 0.0,
            n_t = 0,
            n_tt = 4,
            n_loadstore = 0,
        )
    end
    -log(p)
end
function matching_time_edge(::NaturalAllAtOnce, model::NoiseModel,
                            is_x::Bool, m_dist::Int, first_layer::Bool)
    # Estimate of error on ancilla qubit
    # Same as TypicalSyndrome
    p = calculate_qubit_error_single_pauli(model,
        t_t = 4*model.dur_tt + (is_x ? 2*model.dur_t : 0),
        n_t = (is_x ? 2 : 0),
        n_tt = 4,
        n_meas = 1,
    )
    -log(p)
end
function simulation_noise_parameters(::NaturalAllAtOnce, model::NoiseModel,
                                     ctx::CodeDistanceSim)
    t_round = 4*model.dur_tt + 2*model.dur_t + model.dur_meas
    t_t_data = t_round
    t_t_anc_z = 4*model.dur_tt
    t_t_anc_x = 4*model.dur_tt + 2*model.dur_t
    t_c1_data = ((model.cavity_depth-1)
                 * (ctx.m_dist * t_round + 2*model.dur_loadstore))
    Dict{Symbol, Float64}(
        :p_data_layer1 => combine_error_probs(
            coherence_error(model.t1_c, t_c1_data),
            model.p_loadstore,
            model.p_loadstore,
            coherence_error(model.t1_t, t_t_data)),
        :p_data => coherence_error(model.t1_t, t_t_data),
        :p_anc_z => combine_error_probs(
            calculate_qubit_error_single_pauli(model,
                t_t = t_t_anc_z,
                n_t = 0),
            model.p_meas),
        :p_anc_x => combine_error_probs(
            calculate_qubit_error_single_pauli(model,
                t_t = t_t_anc_x,
                n_t = 2),
            model.p_meas),
        :p_cnot1 => model.p_tt,
        :p_cnot => model.p_tt,
    )
end

# NaturalInterleaved
# Z: One CNOT per data, then measure
# X: Hadamard, one CNOT (reverse dir) per data, hadamard, measure
# Extra store-load error in *every* layer and stored for (cavity-1) layers
function matching_space_edge(::NaturalInterleaved, model::NoiseModel,
                             is_x::Bool, m_dist::Int, first_layer::Bool)
    # Estimate of error on data qubit
    t_round = 4*model.dur_tt + 2*model.dur_t + model.dur_meas
    p = calculate_qubit_error_single_pauli(model,
        t_t = 4*model.dur_tt,
        t_c = ((model.cavity_depth-1) * (t_round + 2*model.dur_loadstore)
               + 2*model.dur_t + model.dur_meas),
        n_t = 0,
        n_tt = 4,
        n_loadstore = 2,
    )
    -log(p)
end
function matching_time_edge(::NaturalInterleaved, model::NoiseModel,
                            is_x::Bool, m_dist::Int, first_layer::Bool)
    # Estimate of error on ancilla qubit
    # Same as TypicalSyndrome
    p = calculate_qubit_error_single_pauli(model,
        t_t = 4*model.dur_tt + (is_x ? 2*model.dur_t : 0),
        n_t = (is_x ? 2 : 0),
        n_tt = 4,
        n_meas = 1,
    )
    -log(p)
end
function simulation_noise_parameters(::NaturalInterleaved, model::NoiseModel,
                                     ctx::CodeDistanceSim)
    t_round = 4*model.dur_tt + 2*model.dur_t + model.dur_meas
    t_t_data = 4*model.dur_tt
    t_t_anc_z = 4*model.dur_tt
    t_t_anc_x = 4*model.dur_tt + 2*model.dur_t
    t_c1_data = ((model.cavity_depth-1) * (t_round + 2*model.dur_loadstore)
                 + 2*model.dur_t + model.dur_meas)
    p_data = combine_error_probs(
        coherence_error(model.t1_c, t_c1_data),
        model.p_loadstore,
        model.p_loadstore,
        coherence_error(model.t1_t, t_t_data))
    Dict{Symbol, Float64}(
        :p_data_layer1 => p_data,
        :p_data => p_data,
        :p_anc_z => combine_error_probs(
            calculate_qubit_error_single_pauli(model,
                t_t = t_t_anc_z,
                n_t = 0),
            model.p_meas),
        :p_anc_x => combine_error_probs(
            calculate_qubit_error_single_pauli(model,
                t_t = t_t_anc_x,
                n_t = 2),
            model.p_meas),
        :p_cnot1 => model.p_tt,
        :p_cnot => model.p_tt,
    )
end

# CompactAllAtOnce
# CNOTs are staggered and interleaved to share qubits between data and ancilla
# so take almost twice as long per layer
# Z: One CNOT per data, then measure
# X: Hadamard, one CNOT (reverse dir) per data, hadamard, measure
# Extra store-load error each layer and stored for (cavity-1)*d layers
function matching_space_edge(::CompactAllAtOnce, model::NoiseModel,
                             is_x::Bool, m_dist::Int, first_layer::Bool)
    # Estimate of error on data qubit
    t_round = (6*model.dur_tt + model.dur_tc + 2*model.dur_t + model.dur_meas
               + 2*model.dur_loadstore)
    p = calculate_qubit_error_single_pauli(model,
            t_t = 3*model.dur_tt,
            t_c = ((first_layer
                        ? (model.cavity_depth-1) * (m_dist * t_round)
                        : 0)
                   + (t_round - 3*model.dur_tt)),
            n_t = 0,
            n_tt = 3,
            n_tc = 1,
            n_loadstore = 2,
        )
    -log(p)
end
function matching_time_edge(::CompactAllAtOnce, model::NoiseModel,
                            is_x::Bool, m_dist::Int, first_layer::Bool)
    # Estimate of error on ancilla qubit
    # One CNOT is transmon-to-cavity
    p = calculate_qubit_error_single_pauli(model,
        t_t = 3*model.dur_tt + model.dur_tc + (is_x ? 2*model.dur_t : 0),
        n_t = (is_x ? 2 : 0),
        n_tt = 3,
        n_tc = 1,
        n_meas = 1,
    )
    -log(p)
end
function simulation_noise_parameters(::CompactAllAtOnce, model::NoiseModel,
                                     ctx::CodeDistanceSim)
    t_round = (6*model.dur_tt + model.dur_tc + 2*model.dur_t + model.dur_meas
               + 2*model.dur_loadstore)
    t_t_data = 3*model.dur_tt
    t_t_anc_z = 3*model.dur_tt + model.dur_tc
    t_t_anc_x = 3*model.dur_tt + model.dur_tc + 2*model.dur_t
    t_c_data = t_round - 3*model.dur_tt
    t_c1_data = t_c_data + (model.cavity_depth-1) * (ctx.m_dist * t_round)
    Dict{Symbol, Float64}(
        :p_data_layer1 => combine_error_probs(
            coherence_error(model.t1_c, t_c1_data),
            model.p_loadstore,
            model.p_loadstore,
            coherence_error(model.t1_t, t_t_data)),
        :p_data => combine_error_probs(
            coherence_error(model.t1_c, t_c_data),
            model.p_loadstore,
            model.p_loadstore,
            coherence_error(model.t1_t, t_t_data)),
        :p_anc_z => combine_error_probs(
            calculate_qubit_error_single_pauli(model,
                t_t = t_t_anc_z,
                n_t = 0),
            model.p_meas),
        :p_anc_x => combine_error_probs(
            calculate_qubit_error_single_pauli(model,
                t_t = t_t_anc_x,
                n_t = 2),
            model.p_meas),
        :p_cnot1 => model.p_tc,
        :p_cnot => model.p_tt,
    )
end

# CompactInterleaved
# CNOTs are staggered and interleaved to share qubits between data and ancilla
# so take almost twice as long per layer
# Z: One CNOT per data, then measure
# X: Hadamard, one CNOT (reverse dir) per data, hadamard, measure
# Extra store-load error each layer and stored for (cavity-1) layers between
function matching_space_edge(::CompactInterleaved, model::NoiseModel,
                             is_x::Bool, m_dist::Int, first_layer::Bool)
    # Estimate of error on data qubit
    t_round = (6*model.dur_tt + model.dur_tc + 2*model.dur_t + model.dur_meas
               + 2*model.dur_loadstore)
    p = calculate_qubit_error_single_pauli(model,
            t_t = 3*model.dur_tt,
            t_c = (model.cavity_depth-1) * t_round + (t_round - 3*model.dur_tt),
            n_t = 0,
            n_tt = 3,
            n_tc = 1,
            n_loadstore = 2,
        )
    -log(p)
end
function matching_time_edge(::CompactInterleaved, model::NoiseModel,
                            is_x::Bool, m_dist::Int, first_layer::Bool)
    # Estimate of error on ancilla qubit
    # Same as CompactAllAtOnce
    p = calculate_qubit_error_single_pauli(model,
        t_t = 3*model.dur_tt + model.dur_tc + (is_x ? 2*model.dur_t : 0),
        n_t = (is_x ? 2 : 0),
        n_tt = 3,
        n_tc = 1,
        n_meas = 1,
    )
    -log(p)
end
function simulation_noise_parameters(::CompactInterleaved, model::NoiseModel,
                                     ctx::CodeDistanceSim)
    t_round = (6*model.dur_tt + model.dur_tc + 2*model.dur_t + model.dur_meas
               + 2*model.dur_loadstore)
    t_t_data = 3*model.dur_tt
    t_t_anc_z = 3*model.dur_tt + model.dur_tc
    t_t_anc_x = 3*model.dur_tt + model.dur_tc + 2*model.dur_t
    t_c_data = (model.cavity_depth-1) * t_round + (t_round - 3*model.dur_tt)
    p_data = combine_error_probs(
        coherence_error(model.t1_c, t_c_data),
        model.p_loadstore,
        model.p_loadstore,
        coherence_error(model.t1_t, t_t_data))
    Dict{Symbol, Float64}(
        :p_data_layer1 => p_data,
        :p_data => p_data,
        :p_anc_z => combine_error_probs(
            calculate_qubit_error_single_pauli(model,
                t_t = t_t_anc_z,
                n_t = 0),
            model.p_meas),
        :p_anc_x => combine_error_probs(
            calculate_qubit_error_single_pauli(model,
                t_t = t_t_anc_x,
                n_t = 2),
            model.p_meas),
        :p_cnot1 => model.p_tc,
        :p_cnot => model.p_tt,
    )
end
