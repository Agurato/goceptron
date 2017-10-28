package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"time"

	gct "github.com/Agurato/goceptron"
)

func main() {

	// Neural Perceptron consts
	const (
		inputLayersize  = 784
		outputLayersize = 10
	)
	// Neural Perceptron vars
	var (
		p                 gct.Perceptron
		expected          []float64
		hiddenLayersSizes []int
		outputError       float64
		eta               float64
	)

	// Stat vars
	var (
		start   time.Time
		elapsed time.Duration
	)

	// File vars
	var (
		imageNb   uint32
		imagePos  uint32
		imageSize uint32
		pixelPos  uint32

		image []byte
		label []byte
	)

	expected = make([]float64, 10)
	hiddenLayersSizes = []int{100}
	p.Init(inputLayersize, hiddenLayersSizes, outputLayersize)
	eta = 0.3

	// Load image file
	trainImages, err := os.Open("train-images.idx3-ubyte")
	if err != nil {
		panic(err)
	}
	defer trainImages.Close()

	// Load label file
	trainLabels, err := os.Open("train-labels.idx1-ubyte")
	if err != nil {
		panic(err)
	}
	defer trainLabels.Close()

	activation := func(input float64) float64 {
		return 1 / (1 + math.Exp(-input))
	}

	magicNumberImages := make([]byte, 4)
	trainImages.Read(magicNumberImages)
	imageNbImages := make([]byte, 4)
	trainImages.Read(imageNbImages)
	rowNb := make([]byte, 4)
	trainImages.Read(rowNb)
	columnNb := make([]byte, 4)
	trainImages.Read(columnNb)

	if binary.BigEndian.Uint32(magicNumberImages) != 2051 {
		fmt.Fprintln(os.Stderr, "Wrong magic number in 'train-images.idx3-ubyte'")
		os.Exit(1)
	}

	imageNb = binary.BigEndian.Uint32(imageNbImages)

	magicNumberLabels := make([]byte, 4)
	trainLabels.Read(magicNumberLabels)
	imageNbLabels := make([]byte, 4)
	trainLabels.Read(imageNbLabels)

	if binary.BigEndian.Uint32(magicNumberLabels) != 2049 {
		fmt.Fprintln(os.Stderr, "Wrong magic number in 'train-labels.idx1-ubyte'")
		os.Exit(1)
	}

	if imageNb != binary.BigEndian.Uint32(imageNbLabels) {
		fmt.Fprintln(os.Stderr, "Different image number in 'train-labels.idx1-ubyte' and 'train-images.idx3-ubyte'")
		os.Exit(1)
	}

	imageSize = binary.BigEndian.Uint32(rowNb) * binary.BigEndian.Uint32(columnNb)
	image = make([]byte, imageSize)
	label = make([]byte, 1)

	// Main loop to repeat until learning is done
	for iter := 0; iter < 100; iter++ {
		// Time calculated for 60000 learnings
		start = time.Now()

		// Go to the beginning of the files to parse them
		trainImages.Seek(16, 0)
		trainLabels.Seek(8, 0)

		outputError = 0

		// For each image
		for imagePos = 0; imagePos < imageNb; imagePos++ {
			pixelPos = 0

			trainImages.Read(image)
			// For each pixel
			for _, pixel := range image {
				p.Layers[0].Neurons[pixelPos].Value = float64(pixel) / 255
				pixelPos++
			}

			trainLabels.Read(label)
			expectedValue := label[0]
			expected[expectedValue] = 1

			p.ComputeFromInputActivation(activation)
			outputError += p.Backpropagation(expected, eta)

			expected[expectedValue] = 0
			if imagePos%1000 == 0 {
				fmt.Printf("\r%d: Image n°%d", iter, imagePos)
			}
		}

		elapsed = time.Since(start)
		fmt.Printf("\r%d: %f (%s)\n", iter, outputError/60000, elapsed)
	}
}
