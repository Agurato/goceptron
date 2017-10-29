package goceptron

import (
	"math"
	"testing"
)

func TestInit(t *testing.T) {
	var p Perceptron

	p.Init(10, []int{}, 10)
	if p.LayerNb != 2 {
		t.Error(p.LayerNb, "Layers, expected 2")
	}

	p = Perceptron{}
	p.Init(10, []int{10}, 10)

	for il, l := range p.Layers {
		if l.Position != p.LayerNb-1 {
			max := 1 / float64(l.Size+1)
			for in, n := range l.Neurons {
				for iw, w := range n.Weights {
					if w <= 0 || w > max {
						t.Error("Layers[", il, "].Neurons[", in, "].Weights[", iw, "], got ", w, " | max = ", max)
					}
				}
			}
			for ib, b := range l.Biases {
				if b <= 0 || b > max {
					t.Error("Layers[", il, "].Biases[", ib, "], got", b, " | max =", max)
				}
			}
		}
	}
}

func TestAddLayer(t *testing.T) {
	var p Perceptron
	p.AddLayer(0)
	if p.LayerNb != 0 {
		t.Error("p.LayerNb =", p.LayerNb, ", expected 0")
	}

	p.AddLayer(10)
	if p.LayerNb != 1 {
		t.Error("p.LayerNb =", p.LayerNb, ", expected 1")
	}
	if p.Layers[0].Size != 10 {
		t.Error("p.Layers[0].Size =", p.Layers[0].Size, ", expected 10")
	}
	if p.Layers[0].Position != 0 {
		t.Error("p.Layers[0].Position =", p.Layers[0].Position, ", expected 0")
	}
	if len(p.Layers[0].Neurons) != 10 {
		t.Error("len(p.Layers[0].Neurons) =", len(p.Layers[0].Neurons), ", expected 10")
	}
	if len(p.Layers[0].Biases) != 0 {
		t.Error("len(p.Layers[0].Biases) =", len(p.Layers[0].Biases), ", expected 0")
	}

	p.AddLayer(7)
	if len(p.Layers[0].Biases) != 7 {
		t.Error("len(p.Layers[0].Biases) =", len(p.Layers[0].Biases), ", expected 7")
	}
	if len(p.Layers[1].Biases) != 0 {
		t.Error("len(p.Layers[1].Biases) =", len(p.Layers[1].Biases), ", expected 0")
	}
}

func TestCalculateLayer(t *testing.T) {
	var p Perceptron

	p.AddLayer(2)
	p.AddLayer(2)

	p.Layers[0].Biases = make([]float64, 2)
	p.Layers[0].Biases[0] = 0.25
	p.Layers[0].Biases[1] = 0.07
	p.Layers[0].Neurons[0].Value = 0.2
	p.Layers[0].Neurons[0].Weights = []float64{0.1, 0.2}
	p.Layers[0].Neurons[1].Value = 0.7
	p.Layers[0].Neurons[1].Weights = []float64{0.2, 0.3}

	p.CalculateLayer(1)

	if p.Layers[1].Neurons[0].Value != 1/(1+math.Exp(-(0.2*0.1+0.7*0.2+0.25))) {
		t.Error("p.Layers[1].Neurons[0].Value =", p.Layers[1].Neurons[0].Value)
	}
	if p.Layers[1].Neurons[1].Value != 1/(1+math.Exp(-(0.2*0.2+0.7*0.3+0.07))) {
		t.Error("p.Layers[1].Neurons[1].Value =", p.Layers[1].Neurons[1].Value)
	}
}

func TestCalculateLayerActivation(t *testing.T) {
	var p Perceptron

	activation := func(input float64) float64 {
		if input < 0.5 {
			return 0
		}
		return 1
	}

	p.AddLayer(2)
	p.AddLayer(2)

	p.Layers[0].Biases = make([]float64, 2)
	p.Layers[0].Biases[0] = 0.5
	p.Layers[0].Biases[1] = 0.07
	p.Layers[0].Neurons[0].Value = 0.2
	p.Layers[0].Neurons[0].Weights = []float64{0.1, 0.2}
	p.Layers[0].Neurons[1].Value = 0.7
	p.Layers[0].Neurons[1].Weights = []float64{0.2, 0.3}

	p.CalculateLayerActivation(1, activation)

	if p.Layers[1].Neurons[0].Value != 1 {
		t.Error("p.Layers[1].Neurons[0].Value =", p.Layers[1].Neurons[0].Value)
	}
	if p.Layers[1].Neurons[1].Value != 0 {
		t.Error("p.Layers[1].Neurons[1].Value =", p.Layers[1].Neurons[1].Value)
	}
}

func TestComputeFromInput(t *testing.T) {
	var p Perceptron
	p.Init(3, []int{3}, 3)

	p.Layers[0].Neurons[0].Value = 0.1
	p.Layers[0].Neurons[1].Value = 0.5
	p.Layers[0].Neurons[2].Value = 0.9

	p.ComputeFromInput()

	if p.Layers[2].Neurons[0].Value <= 0 || p.Layers[2].Neurons[0].Value > 1 {
		t.Error("p.Layers[2].Neurons[0].Value =", p.Layers[2].Neurons[0].Value, ", expected in ]0, 1]")
	}
	if p.Layers[2].Neurons[1].Value <= 0 || p.Layers[2].Neurons[1].Value > 1 {
		t.Error("p.Layers[2].Neurons[1].Value =", p.Layers[2].Neurons[1].Value, ", expected in ]0, 1]")
	}
	if p.Layers[2].Neurons[2].Value <= 0 || p.Layers[2].Neurons[2].Value > 1 {
		t.Error("p.Layers[2].Neurons[2].Value =", p.Layers[2].Neurons[2].Value, ", expected in ]0, 1]")
	}
}

func TestComputeFromInputActivation(t *testing.T) {
	var p Perceptron
	p.Init(3, []int{3}, 3)

	p.Layers[0].Neurons[0].Value = 0.1
	p.Layers[0].Neurons[1].Value = 0.5
	p.Layers[0].Neurons[2].Value = 0.9

	activation := func(input float64) float64 {
		if input < 0.5 {
			return 0
		}
		return 1
	}

	p.ComputeFromInputActivation(activation)

	if p.Layers[2].Neurons[0].Value != 0 && p.Layers[2].Neurons[0].Value != 1 {
		t.Error("p.Layers[2].Neurons[0].Value =", p.Layers[2].Neurons[0].Value, ", expected either 0 or 1")
	}
	if p.Layers[2].Neurons[1].Value != 0 && p.Layers[2].Neurons[1].Value != 1 {
		t.Error("p.Layers[2].Neurons[1].Value =", p.Layers[2].Neurons[1].Value, ", expected either 0 or 1")
	}
	if p.Layers[2].Neurons[2].Value != 0 && p.Layers[2].Neurons[2].Value != 1 {
		t.Error("p.Layers[2].Neurons[2].Value =", p.Layers[2].Neurons[2].Value, ", expected either 0 or 1")
	}
}

func TestTryRecognition(t *testing.T) {
	var p Perceptron

	p.AddLayer(2)
	p.AddLayer(2)

	p.Layers[0].Biases = make([]float64, 2)
	p.Layers[0].Biases[0] = 0.25
	p.Layers[0].Biases[1] = 0.07
	p.Layers[0].Neurons[0].Value = 0.2
	p.Layers[0].Neurons[0].Weights = []float64{0.1, 0.2}
	p.Layers[0].Neurons[1].Value = 0.7
	p.Layers[0].Neurons[1].Weights = []float64{0.2, 0.3}

	output0 := 1 / (1 + math.Exp(-(0.2*0.1 + 0.7*0.2 + 0.25)))
	output1 := 1 / (1 + math.Exp(-(0.2*0.2 + 0.7*0.3 + 0.07)))
	sum := output0 + output1

	rate := p.TryRecognition(0)
	if rate != output0/sum {
		t.Error("rate =", rate, ", expected", output0/sum)
	}

	rate = p.TryRecognition(1)
	if rate != output1/sum {
		t.Error("rate =", rate, ", expected", output1/sum)
	}
}

func TestSaveLoad(t *testing.T) {
	var p1, p2 Perceptron
	p1.Init(784, []int{100, 100}, 10)
	p1.Layers[0].Neurons[0].Weights[0] = 1

	err := p1.SaveToFile("save.goceptron")
	if err != nil {
		t.Fatal(err)
	}

	err = p2.LoadFromFile("save.goceptron")
	if err != nil {
		t.Fatal(err)
	}

	if p1.Layers[0].Neurons[0].Weights[0] != p2.Layers[0].Neurons[0].Weights[0] {
		t.Error("p2.Layers[0].Neurons[0].Weights[0] =", p2.Layers[0].Neurons[0].Weights[0], ", expected", p1.Layers[0].Neurons[0].Weights[0])
	}
}
