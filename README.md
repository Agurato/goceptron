# Perceptron library in Golang

[![Build Status](https://travis-ci.org/cosiner/flag.svg?branch=master&style=flat)](https://travis-ci.org/Agurato/goceptron)
[![Go Report Card](https://goreportcard.com/badge/github.com/cosiner/flag?style=flat)](https://goreportcard.com/report/github.com/Agurato/goceptron)
[![GoDoc](https://img.shields.io/badge/godoc-reference-blue.svg?style=flat)](https://godoc.org/github.com/Agurato/goceptron)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://tldrlegal.com/license/gnu-general-public-license-v3-(gpl-3))

A library implementing a multi-layer perceptron in Golang

## Quick How-to-Use

### Creation of the network

3 parameters are necessary:

- size of the input layer (int)
- sizes of the hidden layers (slice of int)
- size of the output layer (int)

```golang
var (
    p                 gct.Percetron
    inputLayerSize    int
    hiddenLayersSizes []int
    outputLayerSize   int
)
inputLayerSize = 784
hiddenLayersSizes = []int{100}
outputLayerSize = 10
p.Init(inputLayersize, hiddenLayersSizes, outputLayersize)
```

### Learning

For the forward propagation, you can either implement your own neuron activation function, using `p.ComputeFromInputActivation`, or use `p.ComputeFromInput` to use a sigmoid function.
To make the backpropagation, you give as parameters, the expected values of each output neuron, as well as the learning rate eta.

```golang
var (
    expected   []float64
    eta        float64
    activation func(input float64) float64
    mse        float64
)
activation = func(input float64) float64 {
    return 1 / (1 + math.Exp(-input))
}
expected = make([]float64, 10)
eta = 0.3

// Init input layer here
// Modify expected values

p.ComputeFromInputActivation(activation)
mse = p.Backpropagation(expected, eta)
```

### Testing the neural network

To test the neural network and get the recognition rate, use `TryRecognitionActivation` (to be able to use your own activation function) or `TryRecognition` (to use a sigmoid). The return value is the rate of recognition (between 0 and 1)

```golang
var (
    rate float64
)
rate = p.TryRecognitionActivation(activation)
```

Full example, using the MNIST dataset, is available [here](https://github.com/Agurato/goceptron/blob/master/cmd/gct/main.go).

More documentation [here](https://godoc.org/github.com/Agurato/goceptron).

## TODO

- [ ] More GoDoc