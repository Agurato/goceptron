package main

import (
	"bufio"
	"os"
	"strconv"
	"strings"
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
		expected          [10]float64
		hiddenLayersSizes []int
	)

	// File vars
	var (
		lineNumber    int
		scannerImages *bufio.Scanner
		scannerLabels *bufio.Scanner
	)

	p.InitPerceptron(inputLayerSize, hiddenLayersSizes, outputLayerSize)
	// fmt.Println(p.layers[0])

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

	// Main loop to repeat until learning is done
	// for iter := 0; iter < 2; iter++ {
	scannerImages = bufio.NewScanner(trainImages)
	scannerImages.Split(bufio.ScanLines)
	scannerLabels = bufio.NewScanner(trainLabels)
	scannerLabels.Split(bufio.ScanLines)

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
			p.layers[0].neurons[i].value = float64(pixel) / 255
		}
		expected[expectedValue] = 0
	}
	// p.layers[0].Println()
	// }
}
