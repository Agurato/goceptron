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
		inputLayersize  = 784
		outputLayersize = 10
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
	p.InitPerceptron(inputLayersize, hiddenLayersSizes, outputLayersize)

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
		// Time calculated for 60000 learnings
		start = time.Now()

		// Go to the beginning of the files to parse them
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
			scannerLabels.Scan()
			lineImage := scannerImages.Text()
			expectedValue, _ := strconv.Atoi(scannerLabels.Text())
			expected[expectedValue] = 1

			// For each pixel in the line
			for i, pixelString := range strings.Split(lineImage, " ") {
				pixel, _ := strconv.Atoi(pixelString)
				p.Layers[0].Neurons[i].Value = float64(pixel) / 255
			}

			p.ComputeFromInputActivation(activation)
			// p.Backpropagation(expected, 0.2)
			outputError += p.Backpropagation(expected, 0.3)

			expected[expectedValue] = 0
			if lineNumber%1000 == 0 {
				fmt.Printf("\r%d: Image n°%d", iter, lineNumber)
			}
		}

		elapsed = time.Since(start)
		fmt.Printf("\r%d: %f (%s)\n", iter, outputError/60000, elapsed)
	}
}
