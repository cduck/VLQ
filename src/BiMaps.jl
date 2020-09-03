module BiMaps

import Base: show, length, iterate, getindex, setindex!, haskey, get, get!,
    getkey, delete!, pop!, keys, values, pairs, sizehint!, keytype, valtype,
    copy, copy!

export BiMap, rev, hasval, getval


struct BiMap{K, V} <: AbstractDict{K, V}
    _map::Dict{K, V}
    _rev::Dict{V, K}

    function BiMap{K, V}() where {K, V}
        new(Dict{K, V}(), Dict{V, K}())
    end
    function BiMap{K, V}(forward::Dict{K, V}) where {K, V}
        reverse = Dict(v => k for (k, v) in pairs(forward))
        if length(reverse) < length(forward)
            throw(KeyError("BiMap may not contain duplicate values."))
        end
        new(forward, reverse)
    end
    function BiMap{K, V}(forward::Dict{K, V}, reverse::Dict{V, K}) where {K, V}
        @assert length(forward) == length(reverse)
        new(forward, reverse)
    end
end

function BiMap(forward::Dict{K, V}) where {K, V}
    BiMap{keytype(forward), valtype(forward)}(forward)
end
function BiMap(forward::Dict{K, V}, reverse::Dict{V, K}) where {K, V}
    BiMap{keytype(forward), valtype(forward)}(forward, reverse)
end
BiMap{K, V}(args...) where {K, V} = BiMap{K, V}(Dict{K, V}(args...))
BiMap(args...) = BiMap(Dict(args...))

"""Return a `BiMap` with forward and reverse switched."""
@inline rev(map::BiMap{K, V}) where K where V = BiMap{V, K}(map._rev, map._map)

show(io::IO, ::MIME"text/plain", map::BiMap) = show(io, map)
function show(io::IO, map::BiMap)
    buf = IOBuffer()
    show(buf, map._map)
    flush(buf)
    seek(buf, 4)
    write(io, "BiMap")
    write(io, read(buf, String))
    nothing
end

# Internally used default value
struct _Def end
const _def = _Def()

# New methods
length(m::BiMap) = length(m._map)
iterate(m::BiMap, state...) = iterate(m._map, state...)
getindex(m::BiMap, key) = m._map[key]
function setindex!(m::BiMap, val, key)
    old_val = get(m._map, key, _def)
    if old_val !== _def
        delete!(m._rev, old_val)
    elseif old_val == val
        return m
    end
    haskey(m._rev, val) && throw(KeyError(
        "Value already exists in BiMap: $(val) (key=$(key))"))
    m._map[key] = val
    m._rev[val] = key
    m
end
haskey(m::BiMap, key) = haskey(m._map, key)
get(m::BiMap, key, default) = get(m._map, key, default)
get(f::Function, m::BiMap, key) = get(f, m._map, key)
function get!(m::BiMap, key, default)
    haskey(m._map, key) && return m._map[key]
    m[key] = default
end
function get!(f::Function, m::BiMap, key)
    haskey(m._map, key) && return m._map[key]
    m[key] = f()
end
getkey(m::BiMap, key, default) = getkey(m._map, key, default)
delete!(m::BiMap, key) = (pop!(m, key, _def); m)
function pop!(m::BiMap, key, default)
    haskey(m._map, key) || return default
    pop!(m, key)
end
function pop!(m::BiMap, key)
    val = pop!(m._map, key)
    @assert(key == pop!(m._rev, val, _def), "Inconsistent BiMap state")
    val
end
keys(m::BiMap) = keys(m._map)
values(m::BiMap) = values(m._map)
pairs(m::BiMap) = pairs(m._map)
sizehint!(m::BiMap, n) = (sizehint!(m._map, n); sizehint!(m._rev, n))
keytype(m::BiMap) = keytype(m._map)
valtype(m::BiMap) = valtype(m._map)

copy(m::BiMap{K, V}) where {K, V} = BiMap{K, V}(copy(m._map), copy(m._rev))
copy!(dst::BiMap{K, V}, src::BiMap{K, V}) where {K, V} = (
    copy!(dst._map, src._map); copy!(dst._rev, src._rev); dst)

# New functions
hasval(m::BiMap, key) = haskey(m._rev, key)
getval(m::BiMap, key, default) = getkey(m._rev, key, default)


end
