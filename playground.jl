using ConcreteStructs
using Random
using Statistics
using Zygote

Random.seed!(0)

function onehot(y, classes)
    onehot_y = zeros(Bool, (classes, length(y)))
    for (i, label) in enumerate(y)
        onehot_y[label, i] = true
    end
    return onehot_y
end

function spiraldata(samples, classes)
    X = zeros(samples * classes, 2)
    y = zeros(UInt8, samples * classes)
    for classnum in 1:classes
        ix = samples*(classnum-1)+1:samples*classnum
        r = range(0, 1, samples)
        t = range((classnum - 1) * 4, classnum * 4, samples) .+ randn(samples) * 0.2
        X[ix, :] .= [r .* sin.(t * 2.5) r .* cos.(t * 2.5)]
        y[ix] .= classnum
    end
    return transpose(X), onehot(y, classes)
end

function verticaldata(samples, classes)
    X = zeros(samples * classes, 2)
    y = zeros(UInt8, samples * classes)
    for classnum in 1:classes
        ix = samples*(classnum-1)+1:samples*classnum
        X[ix, :] .= [randn(samples) * 0.1 .+ classnum / 3 randn(samples) * 0.1 .+ 0.5]
        y[ix] .= classnum
    end
    return transpose(X), onehot(y, classes)
end

struct Chain{T<:Union{Tuple,NamedTuple}}
    layers::T
end

Chain(xs...) = Chain(xs)
function Chain(; kw...)
    isempty(kw) && return Chain(())
    return Chain(values(kw))
end

# https://github.com/FluxML/Zygote.jl/issues/1126#issuecomment-981191246
# init=x can disconnect the computational graph and make Zygote lose track of the grad
# https://github.com/FluxML/Flux.jl/pull/1809#issuecomment-1009516145
# afoldl needs an rrule
# https://github.com/EnzymeAD/Enzyme.jl/issues/805
# TODO: keep tabs on this
(c::Chain)(x) = foldl(|>, (x, c.layers...))

@concrete struct Dense
    activation
    weight
    bias
end

function Dense(mapping::Pair{<:Int,<:Int}, activation=identity)
    return Dense(first(mapping), last(mapping), activation)
end

function Dense(in_dims::Int, out_dims::Int, activation=identity)
    weight = 0.5 * randn(out_dims, in_dims)
    bias = zeros(out_dims)
    return Dense(activation, weight, bias)
end

@inline __apply_activation(::typeof(identity), x) = x
@inline __apply_activation(f, x) = f.(x)

@inline function (d::Dense)(x::AbstractVecOrMat)
    return __apply_activation(d.activation, d.weight * x .+ d.bias)
end

relu(x) = ifelse(x < 0, zero(x), x)

function softmax(x; dims=1)
    exps = exp.(x .- maximum(x; dims))
    return exps ./ sum(exps; dims)
end

function xlogy(x, y)
    result = x * log(y)
    return ifelse(iszero(x), zero(result), result)
end

epseltype(x) = eps(float(eltype(x)))

function crossentropy(ŷ, y; dims=1, agg=mean, eps::Real=epseltype(ŷ))
    return agg(.-sum(xlogy.(y, ŷ .+ eps); dims=dims))
end

X, y = spiraldata(100, 3)

model = Chain(
    Dense(2 => 256, relu),
    Dense(256 => 256, relu),
    Dense(256 => 3),
    softmax
)

eta = 0.01
epochs = 8000
for epoch in 1:epochs
    loss, ∇model = withgradient(m -> crossentropy(m(X), y), model)
    epoch % 1000 == 0 && println(loss)
    for i in 1:3
        model.layers[i].weight .-= eta * ∇model[1].layers[i].weight
        model.layers[i].bias .-= eta * ∇model[1].layers[i].bias
    end
end
