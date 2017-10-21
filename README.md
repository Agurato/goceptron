# Perceptron in Golang

## Source images used for learning

I used the MNIST Image Dataset to train this perceptron (28*28 pixels in grayscale images) : [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/).

However, since these images are encoded in a weird format, I used this [Github repository](https://github.com/afrozenator/mnist-parser) where the images are already decoded in easily readable format. I modified `train_images.txt` and `train_labels.txt` files to remove all trailing spaces at the end of the lines, as well as changing all line format to CRLF, and re-compressed them in `gzip` format.

Note that only the compressed versions are uploaded here. You have to decompress the files (`train_images.txt.gz` and `train_labels.txt.gz`) to use this program.

## TODO

- [ ] Bias in the weighted sum
