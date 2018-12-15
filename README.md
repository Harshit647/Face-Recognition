# Face-Recognition
This project is a small example of how can one use deep learning libraries like dlib and use them to recognise the unknown images once appropriate training dataset is provided.
Convolutional Neural Networks allow us to extract a wide range of features from images.
Turns out, we can use this idea of feature extraction for face recognition too!
That’s what we are going to explore in this tutorial, using deep conv nets for face recognition.
Note: this is face recognition (i.e. actually telling whose face it is), not just detection (i.e. identifying faces in a picture).

### Idea boiled down :
The approach we are going to use for face recognition is fairly straight forward. 
The key here is to get a deep neural network to produce a bunch of numbers that describe a face (known as face encodings).
When you pass in two different images of the same person, the network should return similar outputs (i.e. closer numbers) for both images, whereas when you pass in images of two different people, the network should return very different outputs for the two images.
This means that the neural network needs to be trained to automatically identify different features of faces and calculate numbers based on that.
The output of the neural network can be thought of as an identifier for a particular person’s face — if you pass in different images of the same person, the output of the neural network will be very similar/close, whereas if you pass in images of a different person, the output will be very different.

Thanks to the developers of dlib library, we will directly use the functions out of it to train our model and then predict the output.

 ## Here are the steps we will be taking:
 

   1. Detect/identify faces in an image (using a face detection model) — for simplicity, this tutorial will only use images with one face/person in it, not more/less

   2. Predict face poses/landmarks (for the faces identified in step 1)

   3. Using data from step 2 and the actual image, calculate face encodings (numbers that describe the face)

   4. Compare the face encodings of known faces with those from test images to tell who is in the picture

## Preparing Images

Firstly, create a project folder (just a folder in which we will keep our code and images). For me it’s called face_recognition but you can call it whatever you like. Inside that folder, create another folder called images . This is the folder that will hold images of the different people you want to run face recognition on. Download some pictures of your friends (one picture per person) from Facebook, rename the picture to your friend’s name (e.g. taus.jpg or john.jpg ) and store all of them in this images folder you just created. One important thing to remember: please make sure that all of those images only have ONE face in them (i.e. they can’t be group pictures) and they are all in JPEG format with filenames ending in .jpg.

Next, create another folder inside your project folder (the face_recognition folder for me) and name it test . This folder will contain different images of the same people whose pictures you stored in the images folder. Again, make sure that each picture only has one person in it. In the test folder, you can name the image files whatever you like and you can have multiple pictures of each person (because we will run face recognition on all pictures in the test folder).

## Installing Dependencies

The most important dependencies for this project are Python 2.7 and pip. You can install both (if you don’t have it already) using Anaconda 2 (which is just a Python distribution that comes pre-packaged with pip) by following this link. Note: Please make sure that Anaconda 2 is added to your PATH and that it’s registered as your system Python 2.7 (there should be a prompt regarding this during the installation process; just press Yes or check the checkbox).

If you are done setting up Anaconda 2 or if you had Python 2.7 and pip installed on your machine beforehand, you can go ahead and install dlib (the machine learning library we will be using) and other dependencies. To do so, type in the following command in Terminal (Mac OS or Linux) or Command Prompt (Windows):

```
pip install --user numpy scipy dlib
```

If there are permission issues, then use the following command:

```
sudo pip install --user numpy scipy dlib
```

## Downloading the pre-trained models

One last thing you need to do is download the pre-trained models for face recognition. There are two models that you need. One model predicts the shape/pose of a face (basically gives you numbers on how the shape is positioned in the image). The other model, takes faces and gives you face encodings (basically numbers that describe the face of that particular person). Here are instructions on how to download, extract and prepare them for our purpose:

1. Download dlib_face_recognition_resnet_model_v1.dat.bz2 from [this link](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2) and shape_predictor_68_face_landmarks.dat.bz2 from [this link](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).
2. Extract the two files in the same folder that you earlier created for the project, in my case it would be 'face_recognition' folder.

## Running the CODE

- Create a new recognize.py file in the 'face_recognition' folder and copy the code from this github project into it.
- run the following command in the terminal:

```
cd {PROJECT_FOLDER_PATH}
python recognize.py
```
The output would look like this:

```
('test/vp.jpg', 'vp')
('test/garvit.jpg', 'garvit')
('test/26850798_1076862052453459_6488511289699706541_o.jpg', 'garvit')
('test/gaurav.jpg', 'gaurav')
('test/26962530_1849859155078320_8057701478832828416_o.jpg', 'garvit')
('test/gagan.jpg', 'gagan')
('test/21543869_1290160577773437_3165857376443869524_o.jpg', 'gaurav')
('test/26850341_1849997201731182_623366753057769405_o.jpg', 'garvit')

```

The name beside the filename shows the name of the person with whom the given face has matched. 
Note that this might not work too well on all images. For optimum performance with this code, try using images that have the face of the person clearly visible. Of course there are other ways of making it accurate (like by actually changing our code to check against multiple images or using jitters, etc.) but the point of this is to just give you a basic idea of how face recognition works.
