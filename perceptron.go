package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// Neuron struct, contains the neuron value, and the weights coming out frmo this neuron
type Neuron struct {
	value   float64
	weights []float64
}

// Layer struct, contains the position of the layer in the perceptron, the number of neurons in it, and the list of the neurons in the layer
type Layer struct {
	position int
	size     int
	neurons  []Neuron
	bias     []float64
}

// Perceptron struct, contains the number of layers, and the list of layers in the perceptron
type Perceptron struct {
	layerNb int
	layers  []Layer
}

// InitPerceptron initializes the neural perceptron
func (p *Perceptron) InitPerceptron(inputLayerSize int, hiddenLayersSizes []int, outputLayerSize int) {
	// Create layers
	p.AddLayer(inputLayerSize)
	for _, size := range hiddenLayersSizes {
		p.AddLayer(size)
	}
	p.AddLayer(outputLayerSize)

	randSource := rand.NewSource(time.Now().UnixNano())
	rand := rand.New(randSource)

	// Create weights
	for il, l := range p.layers {
		if l.position != p.layerNb-1 {
			nextLayerSize := p.layers[il+1].size
			for in := range l.neurons {
				p.layers[il].neurons[in].weights = make([]float64, nextLayerSize)
				for iw := range p.layers[il].neurons[in].weights {
					p.layers[il].neurons[in].weights[iw] = rand.Float64() / float64(l.size)
				}
			}
		}
	}

	for il := range p.layers {
		p.layers[il].position = 10
	}
}

// AddLayer adds a layer containing <size> neurons to the Perceptron
func (p *Perceptron) AddLayer(size int) {
	if size > 0 {
		neurons := make([]Neuron, size)
		p.layers = append(p.layers, Layer{p.layerNb, size, neurons, []float64{}})
		p.layerNb++
	}
	if p.layerNb > 1 {
		p.layers[p.layerNb-2].bias = make([]float64, p.layers[p.layerNb-1].size)
	}
}

// CalculateLayer calculates the new neuron values of the layer which have the position <layerPos> in the perceptron
// TODO: Use bias
func (p *Perceptron) CalculateLayer(layerPos int) {
	if layerPos > 0 {
		sum := make([]float64, p.layers[layerPos].size)
		for in := range p.layers[layerPos].neurons {
			for _, pn := range p.layers[layerPos-1].neurons {
				sum[in] += pn.value * pn.weights[in]
			}
			p.layers[layerPos].neurons[in].value = 1 / (1 + math.Exp(-sum[in]))
		}
	}
}

// ComputeFromInput computes new neuron values, except for the first layer
func (p *Perceptron) ComputeFromInput() {
	for i := 1; i < p.layerNb; i++ {
		p.CalculateLayer(i)
	}
}

// Println prints neuron values in layer
func (l Layer) Println() {
	values := make([]float64, l.size)
	for i, n := range l.neurons {
		values[i] = n.value
	}
	fmt.Println(values)
}
