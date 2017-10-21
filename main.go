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
		inputLayers [784]float64
		lineNumber  int
	)

	// Load images in arrays
	trainImages, err := os.Open("train_images.txt")
	if err != nil {
		panic(err)
	}

	scanner := bufio.NewScanner(trainImages)
	scanner.Split(bufio.ScanLines)

	// For each line
	for scanner.Scan() {
		line := scanner.Text()
		// For each pixel in the line
		if lineNumber == 0 {
			for i, pixelString := range strings.Split(line, " ") {
				pixel, _ := strconv.Atoi(pixelString)
				inputLayers[i] = float64(pixel) / 255
			}
		}
		lineNumber++
	}
	trainImages.Close()

	fmt.Println(inputLayers)
}
