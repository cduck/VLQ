module EvaluateQec

using Random
using LightGraphs
using BlossomV

using ChpSim
using Tweaks.BiMaps, Tweaks.BaseTweaks

export x_gate!, z_gate!, reset!, measure_reset!,
    NoiseModel, SyndromeCircuit, BasicSyndrome,
    CodeDistanceSim, CodeDistanceRun,
    MatchingGraphWeights, apply_sim_error!, do_single_run, do_n_runs


const rng = Random.GLOBAL_RNG


# Extra ChpSim methods
function x_gate!(state::ChpState, qubit::Int)
    hadamard!(state, qubit)
    phase!(state, qubit)
    phase!(state, qubit)
    hadamard!(state, qubit)
end
function z_gate!(state::ChpState, qubit::Int)
    phase!(state, qubit)
    phase!(state, qubit)
end
function reset!(state::ChpState, qubit::Int)
    if measure!(state, qubit, bias=0).value
        x_gate!(state, qubit)
    end
    nothing
end
function measure_reset!(state::ChpState, qubit::Int)
    meas = measure!(state, qubit)
    if meas.value
        x_gate!(state, qubit)
    end
    meas
end


"""
    NoiseModel(...)

Holds probabilities of various quantum error sources
"""
struct NoiseModel
    errors::Dict{Symbol, Float64}
end

"""
    SyndromeCircuit

Parent type representing a style of syndrome measurement circuit.
"""
abstract type SyndromeCircuit end
"""
    BasicSyndrome

Basic syndrome measurement with simple errors
"""
struct BasicSyndrome <: SyndromeCircuit end


NodeT = Tuple{Int, Symbol, Int, Int}
PlaqInfoT = NamedTuple{(:ancilla, :data), Tuple{Int, Vector{Int}}}

"""
    CodeDistanceSim(z_dist[, x_dist, m_dist], syndrome_circuit, noise_model)

Stores the configuration and precomputed data used for error correction.
"""
struct CodeDistanceSim
    z_dist::Int
    x_dist::Int
    m_dist::Int
    syndrome_circuit::SyndromeCircuit
    noise_model::NoiseModel
    num_qubits::Int
    anc_qubits::Vector{Int}
    data_qubits::Vector{Int}
    z_plaqs::Vector{Tuple{Int, Int}}
    z_plaq_info::Vector{PlaqInfoT}
    z_space_boundary::Vector{Tuple{Int, Int}}
    z_graph_nodes::BiMap{NodeT, Int}
    z_graph::Graph{Int}
    z_costs::Matrix{Float64}
    z_bpaths::Set{Tuple{Int, Int}}
    z_path_lengths::Dict{Tuple{Int, Int}, Int}
    x_plaqs::Vector{Tuple{Int, Int}}
    x_plaq_info::Vector{PlaqInfoT}
    x_space_boundary::Vector{Tuple{Int, Int}}
    x_graph_nodes::BiMap{NodeT, Int}
    x_graph::Graph{Int}
    x_costs::Matrix{Float64}
    x_bpaths::Set{Tuple{Int, Int}}
    x_path_lengths::Dict{Tuple{Int, Int}, Int}
end

function CodeDistanceSim(dist::Int, syndrome_circuit::SyndromeCircuit,
                         noise_model::NoiseModel)
    CodeDistanceSim(dist, dist, dist, syndrome_circuit, noise_model)
end
function CodeDistanceSim(z_dist::Int, x_dist::Int, m_dist::Int,
                         syndrome_circuit::SyndromeCircuit, noise_model::NoiseModel)
    # Graphs
    (z_plaqs, z_space_boundary), (x_plaqs, x_space_boundary) = (
        make_plaqs(z_dist, x_dist))
    num_qubits, anc_qubits, data_qubits, z_plaq_info, x_plaq_info = (
        make_qubit_assignments(z_dist, z_plaqs, x_dist, x_plaqs))
    z_graph_nodes, z_graph = (
        make_graph(m_dist+1, z_plaqs, z_space_boundary, false, false))
    x_graph_nodes, x_graph = (
        make_graph(m_dist+1, x_plaqs, x_space_boundary, false, false))

    z_costs, z_bpaths, z_path_lengths = constuct_graph_costs(
        z_graph_nodes, z_graph, noise_model, syndrome_circuit)
    x_costs, x_bpaths, x_path_lengths = constuct_graph_costs(
        x_graph_nodes, x_graph, noise_model, syndrome_circuit)

    CodeDistanceSim(
        z_dist, x_dist, m_dist, syndrome_circuit, noise_model,
        num_qubits, anc_qubits, data_qubits,
        z_plaqs, z_plaq_info, z_space_boundary, z_graph_nodes, z_graph,
            z_costs, z_bpaths, z_path_lengths,
        x_plaqs, x_plaq_info, x_space_boundary, x_graph_nodes, x_graph,
            x_costs, x_bpaths, x_path_lengths,
    )
end


function make_plaqs(z_dist, x_dist)
    z_plaqs = Tuple{Int, Int}[
        (x, y)
        for x in 1:z_dist-1
        for y in 0:x_dist
        if mod(x+y, 2) == 0
    ]
    z_space_boundary = Tuple{Int, Int}[
        (x, y)
        for (x, y) in z_plaqs
        if x == 1 || x == z_dist-1
    ]
    x_plaqs = Tuple{Int, Int}[
        (x, y)
        for x in 0:z_dist
        for y in 1:x_dist-1
        if mod(x+y, 2) == 1
    ]
    x_space_boundary = Tuple{Int, Int}[
        (x, y)
        for (x, y) in x_plaqs
        if y == 1 || y == x_dist-1
    ]
    (z_plaqs, z_space_boundary), (x_plaqs, x_space_boundary)
end
function make_qubit_assignments(z_dist, z_plaqs, x_dist, x_plaqs)
    counter = Iterators.Stateful(Iterators.countfrom(1))
    data_qubits = Dict{Tuple{Int, Int}, Int}(
        (x, y) => popfirst!(counter)
        for x in 1:z_dist
        for y in 1:x_dist
    )
    anc_qubits = Dict{Tuple{Int, Int}, Int}()
    append!(anc_qubits, xy => popfirst!(counter) for xy in z_plaqs)
    append!(anc_qubits, xy => popfirst!(counter) for xy in x_plaqs)
    num_qubits = length(data_qubits) + length(anc_qubits)
    make_plaq_info(plaqs)::Vector{PlaqInfoT} = [
        begin
            qubits = Int[
                data_qubits[(xx, yy)]
                for (xx, yy) in [(x, y), (x, y+1), (x+1, y), (x+1, y+1)]
                if (xx, yy) in keys(data_qubits)
            ]
            @assert(length(qubits) in (2, 4), "Invalid plaquettes")
            (ancilla=anc_qubits[(x, y)], data=qubits)
        end
        for (x, y) in plaqs
    ]
    z_plaq_info = make_plaq_info(z_plaqs)
    x_plaq_info = make_plaq_info(x_plaqs)
    (num_qubits, collect(values(anc_qubits)), collect(values(data_qubits)),
        z_plaq_info, x_plaq_info)
end
function make_graph(m_dist, plaqs, space_boundary,
                    start_boundary::Bool, end_boundary::Bool)
    graph_nodes = BiMap{NodeT, Int}()
    counter = Iterators.Stateful(Iterators.countfrom(1))
    # Assign node ids
    if start_boundary
        for (x, y) in plaqs
            graph_nodes[(0, :tboundary, x, y)] = popfirst!(counter)
        end
    end
    for t in 1:m_dist
        for (x, y) in plaqs
            graph_nodes[(t, :plaq, x, y)] = popfirst!(counter)
        end
        for (x, y) in space_boundary
            graph_nodes[(t, :sboundary, x, y)] = popfirst!(counter)
        end
    end
    if end_boundary
        for (x, y) in plaqs
            graph_nodes[(m_dist+1, :tboundary, x, y)] = popfirst!(counter)
        end
    end
    # Make graph
    plaq_set = Set(plaqs)
    graph = Graph{Int}(length(graph_nodes))
    # Space edges
    for t in 1:m_dist
        for (x, y) in plaqs
            for (x2, y2) in [(x-1, y-1), (x-1, y+1), (x+1, y-1), (x+1, y+1)]
                (x2, y2) in plaq_set || continue
                add_edge!(graph,
                          graph_nodes[(t, :plaq, x, y)],
                          graph_nodes[(t, :plaq, x2, y2)])
            end
        end
        for (x, y) in space_boundary
            add_edge!(graph,
                      graph_nodes[(t, :plaq, x, y)],
                      graph_nodes[(t, :sboundary, x, y)])
            # Connect boundaries
            if (x, y) != space_boundary[1]
                add_edge!(graph,
                          graph_nodes[(t, :sboundary, x, y)],
                          graph_nodes[(t, :sboundary, space_boundary[1]...)])
            end
        end
    end
    # Time edges
    for t in 1-start_boundary:m_dist-1+end_boundary
        type1 = t == 0 ? :tboundary : :plaq
        type2 = t == m_dist ? :tboundary : :plaq
        for (x, y) in plaqs
            add_edge!(graph,
                      graph_nodes[(t, type1, x, y)],
                      graph_nodes[(t+1, type2, x, y)])
        end
        # Connect boundaries
        if 0 < t < m_dist
            add_edge!(graph,
                      graph_nodes[(t, :sboundary, space_boundary[1]...)],
                      graph_nodes[(t+1, :sboundary, space_boundary[1]...)])
        end
    end
    if start_boundary
        # Connect boundaries
        for (x, y) in plaqs
            add_edge!(graph,
                      graph_nodes[(0, :tboundary, x, y)],
                      graph_nodes[(1, :sboundary, space_boundary[1]...)])
        end
    end
    if end_boundary
        # Connect boundaries
        for (x, y) in plaqs
            add_edge!(graph,
                      graph_nodes[(m_dist+1, :tboundary, x, y)],
                      graph_nodes[(m_dist, :sboundary, space_boundary[1]...)])
        end
    end
    graph_nodes, graph
end


"""
Used by construct_graph_costs().
"""
struct MatchingGraphWeights{SyndromeCircuitT} <: AbstractMatrix{Float64}
    graph_nodes::BiMap{NodeT, Int}
    noise_model::NoiseModel
    syndrome_circuit::SyndromeCircuitT
end
Base.size(w::MatchingGraphWeights) = (l=length(w.graph_nodes); (l, l))
function Base.getindex(w::MatchingGraphWeights, i, j)::Float64
    r = rev(w.graph_nodes)
    n, m = r[i], r[j]
    # Inter-boundary
    (n[2] != :plaq && m[2] != :plaq) && return 0.0
    # Time edge (including boundary)
    n[1] != m[1] && return 1.0
    # Space edge (including boundary)
    return 1.0
end

function constuct_graph_costs(graph_nodes::BiMap{NodeT, Int}, graph::Graph{Int},
                              noise_model::NoiseModel,
                              syndrome_circuit::SyndromeCircuit)
    rev_graph_nodes = rev(graph_nodes)
    weights = MatchingGraphWeights(graph_nodes, noise_model, syndrome_circuit)
    paths = floyd_warshall_shortest_paths(graph, weights)
    boundary_ids = Set{Int}(
        id
        for ((_, kind, _, _), id) in graph_nodes
        if kind != :plaq
    )
    boundary_paths = Set{Tuple{Int, Int}}()
    path_parities = Dict{Tuple{Int, Int}, Int}()
    for i in 1:length(graph_nodes)-1
        for j in i+1:length(graph_nodes)
            #i >= j && continue
            if i in boundary_ids && j in boundary_ids
                push!(boundary_paths, (i, j))
                push!(boundary_paths, (j, i))
                continue
            end
            jj = j
            non_boundary_count = 0
            hits_boundary = false
            while jj != 0
                in_boundary = jj in boundary_ids
                jj2 = paths.parents[i, jj]
                is_time_edge = (
                    !in_boundary && jj2 != 0 && !(jj2 in boundary_ids)
                    && rev_graph_nodes[jj][1] != rev_graph_nodes[jj2][1]
                )
                non_boundary_count += !in_boundary && !is_time_edge
                hits_boundary |= in_boundary
                jj = jj2
            end
            if hits_boundary
                push!(boundary_paths, (i, j))
                push!(boundary_paths, (j, i))
            end
            path_parity = non_boundary_count - !hits_boundary
            if !(i in boundary_ids && j in boundary_ids)
                path_parities[(i, j)] = path_parities[(j, i)] = path_parity
            end
        end
    end
    paths.dists, boundary_paths, path_parities
end


"""
    CodeDistanceRun(ctx)

Contains the per error correction run state.
"""
struct CodeDistanceRun
    ctx::CodeDistanceSim
    state::ChpState
    zx_error_counts::Vector{Int}
    zx_meas_error_counts::Vector{Int}
    z_prev::Vector{Bool}
    x_prev::Vector{Bool}
    z_syndromes::Matrix{Bool}
    x_syndromes::Matrix{Bool}
end
function CodeDistanceRun(ctx::CodeDistanceSim)
    state = ChpState(ctx.num_qubits, bitpack=false)
    zx_error_counts = Int[0, 0]
    zx_meas_error_counts = Int[0, 0]
    z_prev = zeros(Bool, length(ctx.z_plaqs))
    x_prev = zeros(Bool, length(ctx.x_plaqs))
    z_syndromes = Matrix{Bool}(undef, length(z_prev), ctx.m_dist+1)
    x_syndromes = Matrix{Bool}(undef, length(x_prev), ctx.m_dist+1)
    CodeDistanceRun(
        ctx, state, zx_error_counts, zx_meas_error_counts,
        z_prev, x_prev, z_syndromes, x_syndromes
    )
end


function apply_sim_error!(noise_model::Nothing,
                          state::ChpState, zx_counts_out::Vector{Int},
                          kind::Symbol, qubits::NTuple{N, Int} where N)
    # Apply no error when no noise model
end
function apply_sim_error!(noise_model::NoiseModel,
                          state::ChpState, zx_counts_out::Vector{Int},
                          kind::Symbol, qubits::NTuple{N, Int} where N)
    p = noise_model.errors[kind]
    # TODO: rng
    z_count = x_count = 0
    for q in qubits
        r = rand(rng, Float64)
        r >= p && return
        # Apply X, Y, or Z with (1/3)p probability
        if r < (2/3)*p
            z_gate!(state, q)
            z_count += 1
        end
        if r >= (1/3)*p
            x_gate!(state, q)
            x_count += 1
        end
    end
    zx_counts_out[2] += z_count  # Z errors cause X syndromes
    zx_counts_out[1] += x_count
    nothing
end

function exec_syndrome_layer(noise_model::Union{NoiseModel, Nothing},
                             syndrome_circuit::BasicSyndrome,
                             run::CodeDistanceRun,
                             layer_i::Int)
    ctx = run.ctx
    state = run.state
    # Apply errors
    for q in ctx.data_qubits
        apply_sim_error!(noise_model, state, run.zx_error_counts,
                         :uniform_data, (q,))
    end
    # Run circuit
    for info in ctx.z_plaq_info
        anc = info.ancilla
        for dat in info.data
            cnot!(state, dat, anc)
        end
    end
    for info in ctx.x_plaq_info
        anc = info.ancilla
        hadamard!(state, anc)
        for dat in info.data
            cnot!(state, anc, dat)
        end
        hadamard!(state, anc)
    end
    # Apply errors
    for q in ctx.anc_qubits
        apply_sim_error!(noise_model, state, run.zx_meas_error_counts,
                         :uniform_anc, (q,))
    end
    # Measure
    for (i, info) in enumerate(ctx.z_plaq_info)
        meas = measure_reset!(state, info.ancilla).value  # TODO: rng
        run.z_syndromes[i, layer_i] = run.z_prev[i] ⊻ meas
        run.z_prev[i] = meas
    end
    for (i, info) in enumerate(ctx.x_plaq_info)
        meas = measure_reset!(state, info.ancilla).value  # TODO: rng
        run.x_syndromes[i, layer_i] = run.x_prev[i] ⊻ meas
        run.x_prev[i] = meas
    end
    nothing
end

"""
    simulate_syndrome_run(run)

Simulate m_dist rounds of syndrome measurement (with start and end boundaries).
"""
function simulate_syndrome_run(run::CodeDistanceRun)
    simulate_syndrome_run(run.ctx.syndrome_circuit, run)
end
function simulate_syndrome_run(syndrome_circuit::SyndromeCircuit, run::CodeDistanceRun)
    ctx = run.ctx
    reset_all_qubits!(run.state)  # Clean start
    run.zx_error_counts .= 0
    run.zx_meas_error_counts .= 0
    exec_syndrome_layer(nothing, syndrome_circuit, run, 1)
    for i in 1:ctx.m_dist+1
        noise = i == ctx.m_dist+1 ? nothing : ctx.noise_model  # Noise-free end layer
        exec_syndrome_layer(noise, syndrome_circuit, run, i)
    end
    run.z_syndromes, run.x_syndromes
end

function syndromes_to_error_ids(syndromes, plaqs, graph_nodes)
    @assert size(syndromes)[1] == length(plaqs)
    Int[
        graph_nodes[(t, :plaq, plaqs[i]...)]
        for t in axes(syndromes, 2)
        for i in axes(syndromes, 1)
        if syndromes[i, t]
    ]
end

function construct_matching_graph(graph_nodes, costs, extra_boundary::NodeT,
                                  error_ids::Vector{Int})
    node_count = div(length(error_ids) + 1, 2) * 2  # Round up to even
    edge_count = div(node_count * (node_count-1), 2)
    matching = Matching(Float64, node_count, edge_count)
    if node_count > length(error_ids)  # Round up to even
        push!(error_ids, graph_nodes[extra_boundary])
    end
    for i in 1:node_count-1#length(error_locations)-1
        for j in i+1:node_count#length(error_ids)
            c = costs[error_ids[i], error_ids[j]]
            add_edge(matching, i-1, j-1, c)
        end
    end
    matching
end

function count_corrected_errors(matching, error_ids, path_lengths)
    count = 0
    for i in 1:length(error_ids)-1
        j = get_match(matching, i-1) + 1  # get_match is zero-indexed
        i < j || continue  # Count each pair once
        count += path_lengths[error_ids[i], error_ids[j]]
    end
    count
end

function match_and_evaluate_syndromes(plaqs, graph_nodes, space_boundary,
                                      costs, path_lengths,
                                      syndromes, error_count)
    error_ids = syndromes_to_error_ids(syndromes, plaqs, graph_nodes)
    extra_boundary = (1, :sboundary, space_boundary[1]...)  # Doesn't matter which
    matching = construct_matching_graph(graph_nodes, costs, extra_boundary, error_ids)
    solve(matching)
    corrected = count_corrected_errors(matching, error_ids, path_lengths)
    failed = mod(error_count + corrected, 2) != 0
    failed
end

"""
    do_single_run(run, z_only=false)

Simulate and error-correct once.  Returns if there was a Z error, or X error,
and inserted error counts.
"""
function do_single_run(run::CodeDistanceRun, z_only::Bool=false)
    z_syndromes, x_syndromes = simulate_syndrome_run(run)
    z_errors, x_errors = run.zx_error_counts

    ctx = run.ctx
    z_fail = match_and_evaluate_syndromes(
        ctx.z_plaqs, ctx.z_graph_nodes, ctx.z_space_boundary, ctx.z_costs,
        ctx.z_path_lengths, z_syndromes, z_errors)
    if z_only
        x_fail = false
    else
        x_fail = match_and_evaluate_syndromes(
            ctx.x_plaqs, ctx.x_graph_nodes, ctx.x_space_boundary, ctx.x_costs,
            ctx.x_path_lengths, x_syndromes, x_errors)
    end
    z_fail, x_fail, (z_errors, x_errors, run.zx_meas_error_counts...,)
end

"""
    do_n_runs(run, n, z_only=false)

Simulate and error-correct n times.  Returns the error probability.
"""
function do_n_runs(run::CodeDistanceRun, n::Int, z_only::Bool=false)
    fail_count = 0
    if z_only
        for _ in 1:n
            z_fail, = do_single_run(run, true)
            fail_count += z_fail
        end
    else
        for _ in 1:n
            z_fail, x_fail = do_single_run(run)
            fail_count += z_fail || x_fail
        end
    end
    return fail_count / n
end


end
