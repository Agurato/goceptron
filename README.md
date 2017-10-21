# Perceptron in Golang

## Source images used for learning

I used the MNIST Image Dataset to train this perceptron (28*28 pixels in grayscale images) : [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/). However, since these images are encoded in a weird format, I used this [Github repository](https://github.com/afrozenator/mnist-parser) where the images are already decoded in easily readable format.

I modified the train_images.txt file from [afrozenator's repo](https://github.com/afrozenator/mnist-parser) to remove all trailing spaces at the end of the lines, as well as changing all line format to CRLF, and re-compressed it to `train_images.txt.gz`.

Note that only the compressed versions are re-uploaded here. You need to decompress the files (`train_images.txt.gz` and `train_labels.txt.gz`) to use this program.
