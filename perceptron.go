// Package goceptron is a more-or-less complete package to manage a perceptron
package goceptron

import (
	"encoding/gob"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"
)

// Neuron struct, contains the neuron Value, and the Weights coming out frmo this neuron
type Neuron struct {
	Value   float64
	Weights []float64
}

// Layer struct, contains the Position of the layer in the perceptron, the number of Neurons in it, and the list of the Neurons in the layer
type Layer struct {
	Position int
	Size     int
	Neurons  []Neuron
	Biases   []float64
}

// Perceptron struct, contains the number of Layers, and the list of Layers in the perceptron
type Perceptron struct {
	LayerNb int
	Layers  []Layer
}

// Init initializes the perceptron
func (p *Perceptron) Init(inputLayersize int, hiddenLayersSizes []int, outputLayersize int) {
	// Create Layers
	p.AddLayer(inputLayersize)
	for _, Size := range hiddenLayersSizes {
		p.AddLayer(Size)
	}
	p.AddLayer(outputLayersize)

	randSource := rand.NewSource(time.Now().UnixNano())
	rand := rand.New(randSource)

	// Create Weights
	for il, l := range p.Layers {
		// If not the last layer
		if l.Position != p.LayerNb-1 {
			nextLayersize := p.Layers[il+1].Size
			// For each neuron
			for in := range l.Neurons {
				// Create slice of Weights
				p.Layers[il].Neurons[in].Weights = make([]float64, nextLayersize)
				// Initialize each weight
				for iw := range p.Layers[il].Neurons[in].Weights {
					p.Layers[il].Neurons[in].Weights[iw] = rand.Float64() / float64(l.Size+1)
				}
			}
			// Initialize bias
			for ib := 0; ib < nextLayersize; ib++ {
				p.Layers[il].Biases[ib] = rand.Float64() / float64(l.Size+1)
			}
		}
	}

	for il := range p.Layers {
		p.Layers[il].Position = 10
	}
}

// AddLayer adds a layer containing <Size> Neurons to the Perceptron
func (p *Perceptron) AddLayer(Size int) {
	if Size > 0 {
		Neurons := make([]Neuron, Size)
		p.Layers = append(p.Layers, Layer{p.LayerNb, Size, Neurons, []float64{}})
		p.LayerNb++
	}
	if p.LayerNb > 1 {
		p.Layers[p.LayerNb-2].Biases = make([]float64, p.Layers[p.LayerNb-1].Size)
	}
}

// CalculateLayer calculates the new neuron Values of the layer which have the Position <layerPos> in the perceptron
// The activation function is a sigmoid
func (p *Perceptron) CalculateLayer(layerPos int) {
	if layerPos > 0 {
		layer := p.Layers[layerPos]
		prevLayer := p.Layers[layerPos-1]
		sum := make([]float64, layer.Size)
		for in := range layer.Neurons {
			for _, pn := range prevLayer.Neurons {
				sum[in] += pn.Value * pn.Weights[in]
			}
			sum[in] += prevLayer.Biases[in]
			p.Layers[layerPos].Neurons[in].Value = 1 / (1 + math.Exp(-sum[in]))
		}
	}
}

// CalculateLayerActivation calculates the new neuron Values of the layer which have the Position <layerPos> in the perceptron
// The activation function is given as a parameter
func (p *Perceptron) CalculateLayerActivation(layerPos int, fn func(float64) float64) {
	if layerPos > 0 {
		layer := p.Layers[layerPos]
		prevLayer := p.Layers[layerPos-1]
		sum := make([]float64, layer.Size)
		for in := range layer.Neurons {
			for _, pn := range prevLayer.Neurons {
				sum[in] += pn.Value * pn.Weights[in]
			}
			sum[in] += prevLayer.Biases[in]
			p.Layers[layerPos].Neurons[in].Value = fn(sum[in])
		}
	}
}

// ComputeFromInput computes new neuron Values, except for the first layer
// The activation function is a sigmoid
func (p *Perceptron) ComputeFromInput() {
	for i := 1; i < p.LayerNb; i++ {
		p.CalculateLayer(i)
	}
}

// ComputeFromInputActivation computes new neuron Values, except for the first layer
// The activation function is given as a parameter
func (p *Perceptron) ComputeFromInputActivation(fn func(float64) float64) {
	for i := 1; i < p.LayerNb; i++ {
		p.CalculateLayerActivation(i, fn)
	}
}

// Backpropagation makes the perceptron learn by modifying the Weights on all Neurons
func (p *Perceptron) Backpropagation(expected []float64, eta float64) (outputError float64) {
	var (
		diff  float64
		delta [][]float64
	)
	delta = make([][]float64, p.LayerNb-1)

	// Calculate error and delta for the output layer
	delta[p.LayerNb-2] = make([]float64, p.Layers[p.LayerNb-1].Size)
	for in, n := range p.Layers[p.LayerNb-1].Neurons {
		diff = expected[in] - n.Value
		delta[p.LayerNb-2][in] = n.Value * (1 - n.Value) * diff
		outputError += math.Pow(diff, 2)
	}

	// Calculates delta for all hidden Layers
	// delta[il-1] is the delta for the layer with Position il (since the input layer doesn't have a delta)
	for il := p.LayerNb - 2; il > 0; il-- {
		layer := p.Layers[il]
		delta[il-1] = make([]float64, layer.Size)
		for in, n := range layer.Neurons {
			for inn := range p.Layers[il+1].Neurons {
				delta[il-1][in] += n.Weights[inn] * delta[il][inn]
			}
			delta[il-1][in] = n.Value * (1 - n.Value) * delta[il-1][in]
		}
	}

	// Update Weights for hidden & input Layers, as well as Biases
	for il := p.LayerNb - 2; il >= 0; il-- {
		layer := p.Layers[il]
		for in, n := range layer.Neurons {
			for inn := range p.Layers[il+1].Neurons {
				p.Layers[il].Neurons[in].Weights[inn] += eta * delta[il][inn] * n.Value
			}
		}
		for ib := range layer.Biases {
			p.Layers[il].Biases[ib] += eta * delta[il][ib]
		}
	}

	return
}

// SaveToFile saves the perceptron to a given file
func (p Perceptron) SaveToFile(path string) (err error) {
	file, err := os.Create(path)
	if err == nil {
		encoder := gob.NewEncoder(file)
		encoder.Encode(p)
	}
	file.Close()
	return
}

// LoadFromFile loads the perceptron from a given file
func (p *Perceptron) LoadFromFile(path string) (err error) {
	file, err := os.Open(path)
	if err == nil {
		decoder := gob.NewDecoder(file)
		err = decoder.Decode(p)
	}
	file.Close()
	return
}

// Println prints neuron Values in layer
func (l Layer) Println() {
	Values := make([]float64, l.Size)
	for i, n := range l.Neurons {
		Values[i] = n.Value
	}
	fmt.Println(Values)
}
