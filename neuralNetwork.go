package main

import (
	"fmt"
	"math/rand"
	"time"
)

// Neuron struct, contains the neuron value, and the weights coming out frmo this neuron
type Neuron struct {
	value   float64
	weights []float64
}

// Layer struct, contains the position of the layer in the network, the number of neurons in it, and the list of the neurons in the layer
type Layer struct {
	position int
	size     int
	neurons  []Neuron
}

// Network struct, contains the number of layers, and the list of layers in the network
type Network struct {
	layerNb int
	layers  []Layer
}

// InitNetwork initializes the neural network
func (net *Network) InitNetwork(inputLayerSize int, hiddenLayersSizes []int, outputLayerSize int) {
	// Create layers
	net.AddLayer(inputLayerSize)
	for _, size := range hiddenLayersSizes {
		net.AddLayer(size)
	}
	net.AddLayer(outputLayerSize)

	randSource := rand.NewSource(time.Now().UnixNano())
	rand := rand.New(randSource)

	// Create weights
	for il, l := range net.layers {
		if l.position != len(hiddenLayersSizes)+1 {
			nextLayerSize := net.layers[il+1].size
			for in := range l.neurons {
				net.layers[il].neurons[in].weights = make([]float64, nextLayerSize)
				for iw := range net.layers[il].neurons[in].weights {
					net.layers[il].neurons[in].weights[iw] = rand.Float64() / float64(l.size)
				}
			}
		}
	}

	for il := range net.layers {
		net.layers[il].position = 10
	}
}

// AddLayer adds a layer containing <size> neurons to the Network
func (net *Network) AddLayer(size int) {
	neurons := make([]Neuron, size)
	net.layers = append(net.layers, Layer{net.layerNb, size, neurons})
	net.layerNb++
}

// Println prints neuron values in layer
func (l Layer) Println() {
	values := make([]float64, l.size)
	for i, n := range l.neurons {
		values[i] = n.value
	}
	fmt.Println(values)
}
