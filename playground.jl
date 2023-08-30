using Random
using Statistics

Random.seed!(0)

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

    X, y
end

function verticaldata(samples, classes)
    X = zeros(samples * classes, 2)
    y = zeros(UInt8, samples * classes)

    for classnum in 1:classes
        ix = samples*(classnum-1)+1:samples*classnum

        X[ix, :] .= [randn(samples) * 0.1 .+ classnum / 3 randn(samples) * 0.1 .+ 0.5]
        y[ix] .= classnum
    end

    X, y
end

struct Dense{F,M<:AbstractMatrix,B<:AbstractVector}
    weight::M
    bias::B
    σ::F

    function Dense(n_inputs, n_neurons, σ::F=identity) where {F}
        weight = 0.01 * randn(n_inputs, n_neurons)
        bias = zeros(n_neurons)

        new{F,typeof(weight),typeof(bias)}(weight, bias, σ)
    end
end

function fwpass(layer::Dense, X)
    layer.σ.(X * layer.weight .+ layer.bias')
end

relu(x) = ifelse(x < 0, zero(x), x)

function softmax(x; dims=2)
    exps = exp.(x .- maximum(x; dims))
    exps ./ sum(exps; dims)
end

function xlogy(x, y)
    result = x * log(y)
    ifelse(iszero(x), zero(result), result)
end

epseltype(x) = eps(float(eltype(x)))

function crossentropy(ŷ, y; dims=2, agg=mean, eps::Real=epseltype(ŷ))
    agg(.-sum(xlogy.(y, ŷ .+ eps); dims=dims))
end

X, y = spiraldata(100, 3)

layer1 = Dense(2, 3, relu)
output1 = fwpass(layer1, X)

layer2 = Dense(3, 3)
output2 = fwpass(layer2, output1)

soutput2 = softmax(output2)

loss = crossentropy(soutput2, y)
println(loss)
