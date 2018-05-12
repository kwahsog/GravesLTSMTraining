# Experimenting with LTSM RNN.

Code largely from: 
https://github.com/deeplearning4j/dl4j-examples

Goal: To experiment tuning/using DL4J. To enable this, have largely taken code samples from DL4J repo, with the aim to tune/build upon.

## LSTM Hyperparameter Tuning
https://deeplearning4j.org/lstm.html
http://karpathy.github.io/2015/05/21/rnn-effectiveness/

### Ideas: 

-Add benchmarking in to track how long it takes to train.
-Load params from JSON.
-Try optimizing methods as per guide.
-Speed up training - optimize JVM.

### Optimizations:

Tried: Changing JVM options (fill in) - no changes.
Approx run time ~ 45 mins. (1 epoch)

Default DL4J settings:
Approx run time ~ 45 mins. (1 epoch)
Set in pom.xml:
<nd4j.backend>nd4j-native-platform</nd4j.backend>

Changing to CUDA:
Approx run time ~ 17 mins. (1 epoch)
Set in pom.xml:
<nd4j.backend>nd4j-cuda-8.0-platform</nd4j.backend>
https://developer.nvidia.com/cuda-downloads
