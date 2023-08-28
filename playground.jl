using Random
using Statistics

Random.seed!(0)

function spiraldata(points, classes)
    X = zeros(Float64, points * classes, 2)
    y = zeros(UInt8, points * classes)

    for class_number = 1:classes
        ix = points*(class_number-1)+1:points*class_number
        r = range(0, 1, points)
        t = range((class_number - 1) * 4, class_number * 4, points) .+ randn(points) * 0.2

        X[ix, :] .= [r .* sin.(t * 2.5) r .* cos.(t * 2.5)]
        y[ix] .= class_number
    end

    X, y
end

struct Dense{M<:AbstractMatrix,B<:AbstractVector}
    weight::M
    bias::B

    function Dense(n_inputs, n_neurons)
        weight = 0.01 * randn(n_inputs, n_neurons)
        bias = zeros(n_neurons)

        new{typeof(weight),typeof(bias)}(weight, bias)
    end
end

function fwpass(layer::Dense, inputs)
    inputs * layer.weight .+ layer.bias'
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

function crossentropy(ŷ, y; dims=2, eps::Real=epseltype(ŷ))
    mean(.-sum(xlogy.(y, ŷ .+ eps); dims=dims))
end

X, y = spiraldata(100, 3)

layer1 = Dense(2, 3)
output1 = fwpass(layer1, X)
activation_output1 = relu.(output1)

layer2 = Dense(3, 3)
output2 = fwpass(layer2, activation_output1)
activation_output2 = softmax(output2)

loss = crossentropy(activation_output2, y)
println(loss)
