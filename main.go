package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

func main() {
	var (
		lineNumber    int
		scannerImages *bufio.Scanner
		scannerLabels *bufio.Scanner

		net      Network
		expected [10]float64
	)

	net.addLayer(784)

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
LINE:
	for scannerImages.Scan() {
		lineNumber++
		scannerLabels.Scan()
		lineImage := scannerImages.Text()
		expectedValue, _ := strconv.Atoi(scannerLabels.Text())
		expected[expectedValue] = 1
		// For each pixel in the line
		for i, pixelString := range strings.Split(lineImage, " ") {
			pixel, _ := strconv.Atoi(pixelString)
			if i > 783 {
				fmt.Print(lineNumber, " ")
				continue LINE
			}
			net.layers[0].neurons[i].value = float64(pixel) / 255
		}
		expected[expectedValue] = 0
	}
	// net.layers[0].Println()
	// }
}
