# Face-Recognition
This project is a small example of how can one use deep learning libraries like dlib and use them to recognise the unknown images once appropriate training dataset is provided.
Convolutional Neural Networks allow us to extract a wide range of features from images.
Turns out, we can use this idea of feature extraction for face recognition too!
That’s what we are going to explore in this tutorial, using deep conv nets for face recognition.
Note: this is face recognition (i.e. actually telling whose face it is), not just detection (i.e. identifying faces in a picture).

Idea boiled down :
The approach we are going to use for face recognition is fairly straight forward. 
The key here is to get a deep neural network to produce a bunch of numbers that describe a face (known as face encodings).
When you pass in two different images of the same person, the network should return similar outputs (i.e. closer numbers) for both images, whereas when you pass in images of two different people, the network should return very different outputs for the two images.
This means that the neural network needs to be trained to automatically identify different features of faces and calculate numbers based on that.
The output of the neural network can be thought of as an identifier for a particular person’s face — if you pass in different images of the same person, the output of the neural network will be very similar/close, whereas if you pass in images of a different person, the output will be very different.

Thanks to the developers of dlib library, we will directly use the functions out of it to train our model and then predict the output.
