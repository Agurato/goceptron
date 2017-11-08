package main

import (
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"os"
)

func init() {
	image.RegisterFormat("jpeg", "jpeg", jpeg.Decode, jpeg.DecodeConfig)
}

// GrayscaledImage returns an image as byte array, each pixel having been grayscaled
func GrayscaledImage(path string) (array [][]uint8) {
	imgfile, err := os.Open(path)
	if err != nil {
		fmt.Println(path, "not found!")
		os.Exit(1)
	}
	defer imgfile.Close()

	imgCfg, _, err := image.DecodeConfig(imgfile)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	width := imgCfg.Width
	height := imgCfg.Height

	fmt.Println("Width : ", width)
	fmt.Println("Height : ", height)

	imgfile.Seek(0, 0)

	// get the image
	img, _, err := image.Decode(imgfile)

	array = make([][]uint8, height)
	for y := 0; y < height; y++ {
		array[y] = make([]uint8, width)
		for x := 0; x < width; x++ {
			array[y][x] = color.GrayModel.Convert(img.At(x, y)).(color.Gray).Y
		}
	}

	return
}
