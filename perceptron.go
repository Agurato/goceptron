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
	biases   []float64
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
		// If not the last layer
		if l.position != p.layerNb-1 {
			nextLayerSize := p.layers[il+1].size
			// For each neuron
			for in := range l.neurons {
				// Create slice of weights
				p.layers[il].neurons[in].weights = make([]float64, nextLayerSize)
				// Initialize each weight
				for iw := range p.layers[il].neurons[in].weights {
					p.layers[il].neurons[in].weights[iw] = rand.Float64() / float64(l.size+1)
				}
			}
			// Initialize bias
			for ib := 0; ib < nextLayerSize; ib++ {
				p.layers[il].biases[ib] = rand.Float64() / float64(l.size+1)
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
		p.layers[p.layerNb-2].biases = make([]float64, p.layers[p.layerNb-1].size)
	}
}

// CalculateLayer calculates the new neuron values of the layer which have the position <layerPos> in the perceptron
// The activation function is a sigmoid
func (p *Perceptron) CalculateLayer(layerPos int) {
	if layerPos > 0 {
		layer := p.layers[layerPos]
		prevLayer := p.layers[layerPos-1]
		sum := make([]float64, layer.size)
		for in := range layer.neurons {
			for _, pn := range prevLayer.neurons {
				sum[in] += pn.value * pn.weights[in]
			}
			sum[in] += prevLayer.biases[in]
			p.layers[layerPos].neurons[in].value = 1 / (1 + math.Exp(-sum[in]))
		}
	}
}

// CalculateLayerActivation calculates the new neuron values of the layer which have the position <layerPos> in the perceptron
// The activation function is given as a parameter
func (p *Perceptron) CalculateLayerActivation(layerPos int, fn func(float64) float64) {
	if layerPos > 0 {
		layer := p.layers[layerPos]
		prevLayer := p.layers[layerPos-1]
		sum := make([]float64, layer.size)
		for in := range layer.neurons {
			for _, pn := range prevLayer.neurons {
				sum[in] += pn.value * pn.weights[in]
			}
			sum[in] += prevLayer.biases[in]
			p.layers[layerPos].neurons[in].value = fn(sum[in])
		}
	}
}

// ComputeFromInput computes new neuron values, except for the first layer
// The activation function is a sigmoid
func (p *Perceptron) ComputeFromInput() {
	for i := 1; i < p.layerNb; i++ {
		p.CalculateLayer(i)
	}
}

// ComputeFromInputActivation computes new neuron values, except for the first layer
// The activation function is given as a parameter
func (p *Perceptron) ComputeFromInputActivation(fn func(float64) float64) {
	for i := 1; i < p.layerNb; i++ {
		p.CalculateLayerActivation(i, fn)
	}
}

// Backpropagation makes the perceptron learn by modifying the weights on all neurons
func (p *Perceptron) Backpropagation(expected []float64, eta float64) (outputError float64) {
	var (
		diff  float64
		delta [][]float64
	)
	delta = make([][]float64, p.layerNb-1)

	// Calculate error and delta for the output layer
	delta[p.layerNb-2] = make([]float64, p.layers[p.layerNb-1].size)
	for in, n := range p.layers[p.layerNb-1].neurons {
		diff = expected[in] - n.value
		delta[p.layerNb-2][in] = n.value * (1 - n.value) * diff
		outputError += math.Pow(diff, 2)
	}

	// Calculates delta for all hidden layers
	// delta[il-1] is the delta for the layer with position il (since the input layer doesn't have a delta)
	for il := p.layerNb - 2; il > 0; il-- {
		layer := p.layers[il]
		delta[il-1] = make([]float64, layer.size)
		for in, n := range layer.neurons {
			for inn := range p.layers[il+1].neurons {
				delta[il-1][in] += n.weights[inn] * delta[il][inn]
			}
			delta[il-1][in] = n.value * (1 - n.value) * delta[il-1][in]
		}
	}

	// Update weights for hidden & input layers, as well as biases
	for il := p.layerNb - 2; il >= 0; il-- {
		layer := p.layers[il]
		for in, n := range layer.neurons {
			for inn := range p.layers[il+1].neurons {
				p.layers[il].neurons[in].weights[inn] += eta * delta[il][inn] * n.value
			}
		}
		for ib := range layer.biases {
			p.layers[il].biases[ib] += eta * delta[il][ib]
		}
	}

	return
}

// Println prints neuron values in layer
func (l Layer) Println() {
	values := make([]float64, l.size)
	for i, n := range l.neurons {
		values[i] = n.value
	}
	fmt.Println(values)
}
