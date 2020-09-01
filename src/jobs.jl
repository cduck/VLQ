"""
Simple management of distributed jobs for parallel work that doesn't block the
REPL.

# Example
```julia
Jobs.launch_workers(16)

@everywhere function my_calculation(arg1, arg2)
    sleep(arg1)
    return arg2
end

Jobs.run_on_workers(
    "run1", my_calculation,
    [((10-i), i) for i in 1:10]
)

# Fetch any completed results without blocking
Jobs.current_results("run1")
Jobs.current_times("run1")  # Job runtimes
```
"""
module Jobs

using Distributed
using OrderedCollections

export launch_workers, run_on_workers,
    current_results, current_times, current_time_info


capacity = 10000
const jobs = RemoteChannel(()->Channel{Tuple{Any, Function, Any}}(capacity))
const results = RemoteChannel(()->Channel{Pair{Tuple{Any, Any}, Tuple}}(
                    capacity))
const collected = OrderedDict()


function kill_workers(; waitfor=30)
    ws = workers()
    (length(ws) == 1 && 1 in ws) && return
    rmprocs(ws, waitfor=waitfor)
end

function clear_jobs()
    while isready(jobs)
        take!(jobs)
    end
    nothing
end

function take_results()
    while isready(results)
        (job_id, job_args), (
            result, t, bytes, gctime, memallocs) = take!(results)
        collected[job_id][job_args] = (result, t, bytes, gctime, memallocs)
    end
    nothing
end

function launch_workers(n::Integer)
    num_new = n - length(setdiff!(Set(workers()), [1]))
    # Create worker processes
    new_workers = addprocs(num_new)
    length(workers()) == n || error("didn't start the right number of workers")
    1 in workers() && error("main process is listed as a worker")
    nothing
end

function run_on_workers(job_id, job_f, job_args_list)
    job_id in keys(collected) && error("job_id <$job_id> has already been used")

    1 in workers() && error("run launch_workers(n) first")

    # Ensure do_work is defined
    @everywhere function _do_work(jobs, results)
        while isready(jobs)
            job_id, job_f, job_args = take!(jobs)
            # Run the job
            val, t, bytes, gctime, memallocs = @timed job_f(job_args...)
            put!(results,
                 (job_id, job_args) => (val, t, bytes, gctime, memallocs))
        end
        #print("Worker $(myid()) is done.")
    end

    # Make jobs
    collected[job_id] = OrderedDict()
    function make_jobs()
        for job_args in job_args_list
            put!(jobs, (job_id, job_f, job_args))
        end
    end
    #@async
    make_jobs()

    # Start worker tasks
    for pid in workers()
        remote_do(Main._do_work, pid, jobs, results)
    end
    nothing
end

function current_results(job_id)
    take_results()
    OrderedDict(
         k => val
         for (k, (val,)) in collected[job_id]
    )
end

function current_times(job_id)
    take_results()
    OrderedDict(
         k => time
         for (k, (val, time)) in collected[job_id]
    )
end

function current_time_info(job_id)
    take_results()
    OrderedDict(
         k => v[2:end]
         for (k, v) in collected[job_id]
    )
end


end  # module
