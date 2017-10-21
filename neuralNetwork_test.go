package main

import (
	"testing"
)

func TestInitNetwork(t *testing.T) {
	inputLayerSize := 784
	outputLayerSize := 10
	hiddenLayersSizes := []int{50}

	var net Network
	net.InitNetwork(inputLayerSize, hiddenLayersSizes, outputLayerSize)

	for il, l := range net.layers {
		if l.position != len(hiddenLayersSizes)+1 {
			for in, n := range l.neurons {
				for iw, w := range n.weights {
					if w <= 0 || w > 1/float64(l.size) {
						t.Error("layers[", il, "].neurons[", in, "].weights[", iw, "], got ", w, " | max = ", 1/float64(l.size))
					}
				}
			}
		}
	}
}
