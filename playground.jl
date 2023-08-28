using Random

Random.seed!(0)

function spiral_data(points, classes)
    X = zeros(points * classes, 2)
    y = zeros(UInt8, points * classes)

    for class_number in 1:classes
        ix = points * class_number : (points * class_number + 1) - 1
        r = range(0, 1, points)
        t = range(class_number*4, (class_number+1)*4, points) + randn(points) * 0.2

        X[ix, :] = hcat(r * sin(t*2.5), r * cos(t*2.5))
        y[ix] = class_number
    end
    return X, y
end

struct Dense{M<:AbstractMatrix, B<:AbstractVector}
    weight::M
    bias::B

    function Dense(n_inputs, n_neurons)
        weight = 0.01 * randn(n_inputs, n_neurons)
        bias = zeros(n_neurons)

        new{typeof(weight),typeof(bias)}(weight, bias)
    end
end

function forward_pass(layer::Dense, inputs)
    return inputs * layer.weight .+ layer.bias'
end

relu(x) = ifelse(x<0, zero(x), x)

X, y = spiral_data(100, 3)

layer1 = Dense(2, 3)

output1 = forward_pass(layer1, X)
activation_output1 = relu.(output1)

println(activation_output1)
