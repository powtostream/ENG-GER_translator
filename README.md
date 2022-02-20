# ENG-GER_translator
## Neural translation model

For the capstone project, I used a language dataset from http://www.manythings.org/anki/ to build a neural translation model.
This dataset consists of over 200,000 pairs of sentences in English and German.
In order to make the training quicker, we will restrict to our dataset to 80,000 pairs.
The goal is to develop a neural translation model from English to German, making use of a pre-trained English word embedding module.

## The structure

The custom model consists of an encoder RNN and a decoder RNN.
The encoder takes words of an English sentence as input, and uses a pre-trained word embedding to embed the words into a 128-dimensional space.
To indicate the end of the input sentence, a special end token (in the same 128-dimensional space) is passed in as an input.
This token is a TensorFlow Variable that is learned in the training phase (unlike the pre-trained word embedding, which is frozen).

The decoder RNN takes the internal state of the encoder network as its initial state.
A start token is passed in as the first input, which is embedded using a learned German word embedding.
The decoder RNN then makes a prediction for the next German word, which during inference is then passed in as the following input,
and this process is repeated until the special <end> token is emitted from the decoder.

