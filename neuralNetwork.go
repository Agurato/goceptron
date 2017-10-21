package main

import "fmt"

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

// Adds a layer containing <size> neurons to the Network
func (net *Network) addLayer(size int) {
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
