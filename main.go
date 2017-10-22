package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"time"
)

func main() {

	// Neural Perceptron consts
	const (
		inputLayerSize  = 784
		outputLayerSize = 10
	)
	// Neural Perceptron vars
	var (
		p                 Perceptron
		expected          []float64
		hiddenLayersSizes []int
		outputError       float64
	)

	// Stat vars
	var (
		start   time.Time
		elapsed time.Duration
	)

	// File vars
	var (
		lineNumber    int
		scannerImages *bufio.Scanner
		scannerLabels *bufio.Scanner
	)

	expected = make([]float64, 10)
	hiddenLayersSizes = []int{100, 100}
	p.InitPerceptron(inputLayerSize, hiddenLayersSizes, outputLayerSize)

	// Load image file
	trainImages, err := os.Open("train_images.txt")
	if err != nil {
		panic(err)
	}
	defer trainImages.Close()

	// Load label file
	trainLabels, err := os.Open("train_labels.txt")
	if err != nil {
		panic(err)
	}
	defer trainLabels.Close()

	activation := func(input float64) float64 {
		return 1 / (1 + math.Exp(-input))
	}

	// Main loop to repeat until learning is done
	for iter := 0; iter < 100; iter++ {
		start = time.Now()

		trainImages.Seek(0, 0)
		scannerImages = bufio.NewScanner(trainImages)
		scannerImages.Split(bufio.ScanLines)
		trainLabels.Seek(0, 0)
		scannerLabels = bufio.NewScanner(trainLabels)
		scannerLabels.Split(bufio.ScanLines)

		outputError = 0
		lineNumber = 0
		// For each line
		for scannerImages.Scan() {
			lineNumber++
			// if lineNumber < 3 {
			scannerLabels.Scan()
			lineImage := scannerImages.Text()
			expectedValue, _ := strconv.Atoi(scannerLabels.Text())
			expected[expectedValue] = 1

			// For each pixel in the line
			for i, pixelString := range strings.Split(lineImage, " ") {
				pixel, _ := strconv.Atoi(pixelString)
				p.layers[0].neurons[i].value = float64(pixel) / 255
			}

			p.ComputeFromInputActivation(activation)
			// p.Backpropagation(expected, 0.2)
			outputError += p.Backpropagation(expected, 0.3)

			expected[expectedValue] = 0
			// }
		}

		elapsed = time.Since(start)
		fmt.Printf("%d: %f (%s)\n", iter, outputError/60000, elapsed)
	}
}
