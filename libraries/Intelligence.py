import csv
import random


def ran():
    return random.random()


def CreateLayers(layers):
    layerArr = []
    for layer in layers:
        layerArr.append(Layer(layer))
    return layerArr


def clamp(number, min_value, max_value):
    return max(min(number, max_value), min_value)


def flatten(data):
    return [item for sublist in data for item in sublist]


def ReLU(val):
    return clamp(val, 0, 1)


# Normalized Squared Error
def NSE(difference, target):
    return (difference**2) / (target ** 2)


class Neuron:
    def __init__(self):
        self.weight = 0.0

    def predict(self, data):
        result = []
        for x in data:
            result.append(ReLU(float(x) * self.weight))
        return result

    def train(self, error, learn_rate):
        self.weight += learn_rate * error * ran()
        self.weight = clamp(self.weight, -1.0, 1.0)

    def export(self):
        return self.weight


class Layer:
    def __init__(self, size):
        self.neurons = [Neuron() for _ in range(size)]
        self.size = size
        self.bias = 0.0

    def predict(self, data):
        out = []
        for neuron in self.neurons:
            out.extend(neuron.predict(data))
        return [x + self.bias for x in out]

    def train(self, error, learn_rate):
        self.bias += learn_rate * error * ran()
        self.bias = clamp(self.bias, 0, 1)
        for neuron in self.neurons:
            neuron.train(error, learn_rate)

    def export(self):
        out = [self.bias]
        for neuron in self.neurons:
            out.append(neuron.export())
        return out


class InputLayer:
    def __init__(self, size):
        self.neurons = [Neuron() for _ in range(size)]
        self.size = size
        self.bias = 0.0

    def predict(self, data):
        out = []
        for neuron, x in zip(self.neurons, data):
            out.extend(neuron.predict([x]))
        return [x + self.bias for x in out]

    def train(self, error, learn_rate):
        self.bias += learn_rate * error * ran()
        self.bias = clamp(self.bias, 0, 1)
        for neuron in self.neurons:
            neuron.train(error, learn_rate)

    def export(self):
        out = [self.bias]
        for neuron in self.neurons:
            out.append(neuron.export())
        return out


class OutputLayer:
    def __init__(self, size):
        self.neurons = [Neuron() for _ in range(size)]
        self.size = size
        self.bias = 0.0

    def predict(self, data):
        return sum(data) + self.bias

    def train(self, error, learn_rate):
        self.bias += learn_rate * error * ran()
        self.bias = clamp(self.bias, 0, 1)
        for neuron in self.neurons:
            neuron.train(error, learn_rate)

    def export(self):
        out = [self.bias]
        for neuron in self.neurons:
            out.append(neuron.export())
        return out


class Intelligence:
    def __init__(self, input_layer, hidden_layers, output_layer, learning_rate):
        self.input_layer = InputLayer(input_layer)
        self.hidden_layers = CreateLayers(hidden_layers)
        self.output_layer = OutputLayer(output_layer)
        self.learning_rate = learning_rate

    def train(self, data, target):
        output = self.predict(data)
        error = NSE(target - output, target)
        if target > 1 > output or target < 1 < output:
            error *= 5
        if output > target:
            error *= -1
        if abs(error) > 10:
            error = 0
        # if ran() < 0.0003:
            # print(f"target: {target}, error: {error}, prediction: {output}")
        self.input_layer.train(error, self.learning_rate)
        for layer in self.hidden_layers:
            layer.train(error, self.learning_rate)
        self.output_layer.train(error, self.learning_rate)

    def predict(self, data):
        data = self.input_layer.predict(data)
        for layer in self.hidden_layers:
            data = layer.predict(data)
        data = self.output_layer.predict(data)
        return data

    def export(self):
        out = [self.input_layer.export()]
        for layer in self.hidden_layers:
            out.append(layer.export())
        out.append(self.output_layer.export())
        return out


def export_intelligence(intelligence, output_file):
    print(output_file)
    raise ValueError("Success!")
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write input layer
        for neuron in intelligence.input_layer.neurons:
            writer.writerow(['Input', neuron.weight, intelligence.input_layer.bias])

        # Write hidden layers
        for layer_num, layer in enumerate(intelligence.hidden_layers, start=1):
            for neuron in layer.neurons:
                writer.writerow([f'Hidden {layer_num}', neuron.weight, layer.bias])

        # Write output layer
        for neuron in intelligence.output_layer.neurons:
            writer.writerow(['Output', neuron.weight, intelligence.output_layer.bias])