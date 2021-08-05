"""
Generate plots for the paper.

# Example
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
"""
module MakePlots

using Distributed
import PyPlot; plt = PyPlot

export setup, dist_calc_all, fetch_finished, plot, plot_finished

include("vlq.jl")
include("jobs.jl"); using .Jobs

function setup(; num_workers=16)
    Jobs.launch_workers(num_workers)
    eval(:(@everywhere (
        include("src/vlq.jl");
        using .VLQ.BiMaps
    )))

    @everywhere function everywhere_calc_single(
            plot_i, syndrome_sym, d, e, override_key, samples)
        step = 100
        e *= 9 <= plot_i <=11 ? 1.0e9 : 1.0
        if override_key === nothing
            override_pairs = ()
        else
            override_pairs = [(override_key, e)]
            e = VLQ.sensitivity_base_p
        end

        # Shortcut for slow high error runs
        (e > 0.01 && d > 5) && (samples = min(samples, 10000))
        e > 0.03 && (samples = min(samples, 1000))
        e >= 0.1 && (samples = min(samples, 500))

        model = VLQ.make_noise_model_for_paper(e, override_pairs)
        ctx = VLQ.CodeDistanceSim(d,
            getproperty(VLQ, syndrome_sym)(),
            model
        )
        run = VLQ.CodeDistanceRun(ctx)
        loops = div(samples, step)
        sum(let r = VLQ.do_n_runs(run, step, false)
                ccall(:jl_gc_collect, Nothing, ())
                r
            end
            for _ in 1:loops
        ) / loops
    end
end


function x_axis_for_plot(plot_i)
    if 1 <= plot_i <= 5
        10 .^ LinRange(log10(0.1), log10(0.0001), 19)[4:end-2]
    elseif 6 <= plot_i <= 8
        10 .^ LinRange(log10(0.1), log10(1e-5), 13+3)[3:end]
    elseif plot_i == 9
        10 .^ LinRange(log10(1e-5), log10(1), 16)[1:end-3]
    elseif plot_i == 10
        10 .^ LinRange(log10(1e-5), log10(1), 16)[1:end-3]
    elseif plot_i == 11
        10 .^ LinRange(log10(1e-4), log10(1e-8), 17)[1:end-4]
    elseif plot_i == 12
        LinRange(2, 30, 15)
    end
end
function confs_for_plot(plot_i, dists, samples)
    if 1 <= plot_i <= 5
        confs_for_thresh_plot(plot_i, dists, samples)
    else
        confs_for_sens_plot(plot_i, dists, samples)
    end
end
function confs_for_thresh_plot(plot_i, dists, samples)
    x_arr = x_axis_for_plot(plot_i)
    syndrom_sym = [
        :TypicalSyndrome, :NaturalAllAtOnce, :NaturalInterleaved,
        :CompactAllAtOnce, :CompactInterleaved][plot_i]
    [
        (plot_i, syndrom_sym, d, e, nothing, samples)
        for e in x_arr
        for d in dists
    ]
end
function confs_for_sens_plot(plot_i, dists, samples)
    x_arr = x_axis_for_plot(plot_i)
    override_key = [
        :p_tt, :p_loadstore, :p_tc, :t1_c, :t1_t,
        :dur_loadstore, :cavity_depth][plot_i-5]
    [
        (plot_i, :CompactInterleaved, d, e, override_key, samples)
        for e in x_arr
        for d in dists
    ]
end

function dist_calc_all(job_id, dists, samples, which_plots=1:12)
    confs = []
    for i in which_plots
        append!(confs, confs_for_plot(i, dists, samples))
    end
    Jobs.run_on_workers(job_id, Main.everywhere_calc_single, confs)
end

function fetch_finished(job_id, plot_i, default::Float64=1.0)
    x_arr = x_axis_for_plot(plot_i)
    y_arr_dict = Dict{Int, Vector{Float64}}()
    results = Jobs.current_results(job_id)
    for ((res_plot_i, _, d, e, _, _), val) in results
        res_plot_i == plot_i || continue
        if !(d in keys(y_arr_dict))
            y_arr_dict[d] = repeat([default], length(x_arr))
        end
        i = indexin([e], x_arr)[1]
        i === nothing || (y_arr_dict[d][i] = val)
    end
    dists = sort!(collect(keys(y_arr_dict)))
    y_arrs = [y_arr_dict[d] for d in dists]
    x_arr, dists, y_arrs
end


function plot(plot_i, x_arr, dists, y_arrs)
    println("plot_i = $plot_i")
    println("x_arr = $x_arr")
    println("dists = $dists")
    println("y_arrs = $y_arrs")

    fig, ax = plt.subplots(1, 1)
    if plot_i <= 5
        ax.plot(x_arr, x_arr, ":k")
    end
    for (d, y_arr) in zip(dists, y_arrs)
        ax.plot(x_arr, y_arr, label="$d")
    end
    ax.legend()
    if plot_i == 12
        ax.set_yscale("log")
    else
        ax.loglog()
    end
    title = ["Baseline", "NatAll", "NatInt", "CompAll", "CompInt",
             "SC-SC Err", "L/S Err", "SC-Mode Err", "Cav T1",
             "Trans T1", "L/S Dur", "Cav Size"][plot_i]
    ax.set_title(title)
    nothing
end

function plot_finished(job_id, plot_i, default::Float64=1.0)
    x_arr, dists, y_arrs = fetch_finished(job_id, plot_i, default)
    plot(plot_i, x_arr, dists, y_arrs)
end


end
