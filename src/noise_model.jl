export TypicalSyndrome, MeqcNormalAll, MeqcNormalRoundRobin, MeqcCompactAll,
    MeqcCompactRoundRobin,
    make_noise_model_for_paper


# The different circuits evaluated
struct TypicalSyndrome <: AbstractFastSyndrome end
struct MeqcNormalAll <: AbstractFastSyndrome end
struct MeqcNormalRoundRobin <: AbstractFastSyndrome end
struct MeqcCompactAll <: AbstractFastSyndrome end
struct MeqcCompactRoundRobin <: AbstractFastSyndrome end


# Make the noise model used in the paper
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
sensitivity_base_p = 5e-3

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


# Util functions
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

# Typical Syndrome
# Z: One CNOT per data, then measure
# X: Hadamard, one CNOT (reverse dir) per data, hadamard, measure
function _data_ancilla_times_in_transmon_cavity(
        ::TypicalSyndrome, model::NoiseModel, is_x::Bool
        )::NTuple{4, Float64}
    t_cycle = 4*model.dur_tt + 2*model.dur_t + model.dur_meas
    t_z_anc = 4*model.dur_tt
    t_x_anc = 4*model.dur_tt + 2*model.dur_t
    t_anc = is_x ? t_x_anc : t_z_anc
    # Data in transmon, data in cavity, anc in transmon, anc in cavity
    (t_cycle, 0, t_anc, 0)
end
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
function matching_space_edge(::TypicalSyndrome, model::NoiseModel, is_x::Bool)
    # Mainly 4 CNOTs; one per plaquette
    p = calculate_qubit_error_single_pauli(model,
        t_t = 4*model.dur_tt + 2*model.dur_t + model.dur_meas,
        n_t = 0,
        n_tt = 4,
    )
    -log(p)
end
function matching_time_edge(::TypicalSyndrome, model::NoiseModel, is_x::Bool)
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

function simulation_noise_parameters(::TypicalSyndrome, model::NoiseModel)
    t_data = 4*model.dur_tt + 2*model.dur_t + model.dur_meas
    t_anc_z = 4*model.dur_tt
    t_anc_x = 4*model.dur_tt + 2*model.dur_t
    Dict{Symbol, Float64}(
        :p_data => coherence_error(model.t1_t, t_data),
        :p_anc_z => combine_error_probs(
            calculate_qubit_error_single_pauli(model,
                t_t = t_anc_z,
                n_t = 0),
            calculate_qubit_error_single_pauli(model,
                t_t = t_anc_z,
                n_t = 0),
            model.p_meas),
        :p_anc_x => combine_error_probs(
            calculate_qubit_error_single_pauli(model,
                t_t = t_anc_x,
                n_t = 2),
            calculate_qubit_error_single_pauli(model,
                t_t = t_anc_x,
                n_t = 2),
            model.p_meas),
        :p_cnot1 => model.p_tt,
        :p_cnot => model.p_tt,
    )
end
