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
		activation        func(input float64) float64
		derivative        func(input float64) float64
		expectedValue     int
	)

	// Stat vars
	var (
		iter      uint32
		start     time.Time
		elapsed   time.Duration
		correctNb uint32
	)

	// File vars
	var (
		trainImages *os.File
		trainLabels *os.File
		testImages  *os.File
		testLabels  *os.File

		imageNb      uint32
		imagePos     uint32
		imageSize    uint32
		testImagePos uint32

		trainImage []byte
		trainLabel []byte
		testImage  []byte
		testLabel  []byte
	)

	var err error
	const testNumber uint32 = 10000
	const testInterval uint32 = 10000

	expected = make([]float64, 10)
	hiddenLayersSizes = []int{300}
	p.Init(inputLayersize, hiddenLayersSizes, outputLayersize)
	eta = 0.3

	// Load train image file
	trainImages, err = os.Open("train-images.idx3-ubyte")
	if err != nil {
		panic(err)
	}
	defer trainImages.Close()

	// Load train label file
	trainLabels, err = os.Open("train-labels.idx1-ubyte")
	if err != nil {
		panic(err)
	}
	defer trainLabels.Close()

	// Load test image file
	testImages, err = os.Open("t10k-images.idx3-ubyte")
	if err != nil {
		panic(err)
	}
	defer testImages.Close()

	// Load test label file
	testLabels, err = os.Open("t10k-labels.idx1-ubyte")
	if err != nil {
		panic(err)
	}
	defer testLabels.Close()

	imageNb, imageSize, err = checkFiles(trainImages, trainLabels)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	_, _, err = checkFiles(testImages, testLabels)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	trainImage = make([]byte, imageSize)
	trainLabel = make([]byte, 1)
	testImage = make([]byte, imageSize)
	testLabel = make([]byte, 1)

	activation = func(input float64) float64 {
		return 1 / (1 + math.Exp(-input/2))
	}
	derivative = func(input float64) float64 {
		return input * (1 - input)
	}

	start = time.Now()
	// Main loop to repeat until learning is done
	for iter = 0; iter < 5; iter++ {

		// Go to the beginning of the files' data to parse them
		trainImages.Seek(16, 0)
		trainLabels.Seek(8, 0)

		// For each image
		for imagePos = 0; imagePos < imageNb; imagePos++ {
			trainImages.Read(trainImage)
			// For each pixel
			for ipixel, pixel := range trainImage {
				p.Layers[0].Neurons[ipixel].Value = float64(pixel) / 255
			}

			trainLabels.Read(trainLabel)
			expectedValue = int(trainLabel[0])
			expected[expectedValue] = 1

			// Forward propagation
			p.ComputeFromInputCustom(activation)
			// Back-propagation
			outputError += p.BackpropagationCustom(expected, eta, derivative)

			expected[expectedValue] = 0
			if imagePos%testInterval == 0 && (iter*imageNb+imagePos != 0) {
				// eta /= 1.01
				elapsed = time.Since(start)
				fmt.Printf("Image n°%d (%s): eta=%f\n", iter*imageNb+imagePos, elapsed, eta)
				fmt.Printf("\tMean MSE =\t%.10f\n", outputError/float64(testInterval))
				outputError = 0

				testImages.Seek(16, 0)
				testLabels.Seek(8, 0)

				var meanCertainty float64
				correctNb = 0
				for testImagePos = 0; testImagePos < testNumber; testImagePos++ {
					testImages.Read(testImage)
					// For each pixel
					for ipixel, pixel := range testImage {
						p.Layers[0].Neurons[ipixel].Value = float64(pixel) / 255
					}

					testLabels.Read(testLabel)

					recogCertainty, correct := p.TryRecognitionCustom(int(testLabel[0]), activation)
					meanCertainty += recogCertainty
					if correct {
						correctNb++
					}
				}
				meanCertainty /= float64(testNumber)

				fmt.Printf("\tMean certainty = %f %%\n", meanCertainty*100)
				fmt.Printf("\tRecognition rate = %.2f %%\n", float64(correctNb)/float64(testNumber)*100)
			}
		}
	}

	date := time.Now().Local()
	saveFileName := fmt.Sprintf("%04d%02d%02d-%02d%02d%02d.gct", date.Year(), date.Month(), date.Day(), date.Hour(), date.Minute(), date.Second())
	p.SaveToFile(saveFileName)
}

func checkFiles(imagesFile, labelsFile *os.File) (uint32, uint32, error) {
	var (
		imageNb   uint32
		imageSize uint32
	)

	magicNumberImages := make([]byte, 4)
	imagesFile.Read(magicNumberImages)
	imageNbImages := make([]byte, 4)
	imagesFile.Read(imageNbImages)
	rowNb := make([]byte, 4)
	imagesFile.Read(rowNb)
	columnNb := make([]byte, 4)
	imagesFile.Read(columnNb)

	if binary.BigEndian.Uint32(magicNumberImages) != 2051 {
		return 0, 0, fmt.Errorf("Wrong magic number in '%s'", imagesFile.Name())
	}

	imageNb = binary.BigEndian.Uint32(imageNbImages)
	imageSize = binary.BigEndian.Uint32(rowNb) * binary.BigEndian.Uint32(columnNb)

	magicNumberLabels := make([]byte, 4)
	labelsFile.Read(magicNumberLabels)
	imageNbLabels := make([]byte, 4)
	labelsFile.Read(imageNbLabels)

	if binary.BigEndian.Uint32(magicNumberLabels) != 2049 {
		return 0, 0, fmt.Errorf("Wrong magic number in '%s'", labelsFile.Name())
	}

	if imageNb != binary.BigEndian.Uint32(imageNbLabels) {
		return 0, 0, fmt.Errorf("Different number of images in '%s' and '%s'", imagesFile.Name(), labelsFile.Name())
	}

	return imageNb, imageSize, nil
}
