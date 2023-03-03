# An examination of the application of spectrograms for gender identification in vocal analysis.
The data was provided, which contained 1500 voice recordings in .mp3 format. Each recording was converted into a spectrogram and then entered into the desired folder.

Also, a csv table was added to the data, in which it was clarified on which record whose voice.

| Parameter | Quantity  | 
| :------------ |:---------------:|
| Records| 1500 |
| Female voice| 355|
| Male voice | 1145|
| Train data | 664|
| Val data | 664|
| Test data | 172|


A neural network was used as a model  Convolutional Neural Network (CNN). It is a type of neural network commonly used for image classification tasks. The network architecture includes multiple layers of convolutional and pooling layers, followed by a few fully connected layers. The Rescaling layer at the beginning is used to scale the input image pixel values to the range of [0, 1]. The output layer uses a sigmoid activation function to produce a binary classification output. 

Although the data used in the study was unbalanced, the model achieved an accuracy of 77%, possibly due to the absence of extraneous noises in the recordings. Increasing the size of the dataset mayfurther improve the model's performance. Overall, the findings suggest that spectrograms can be used to determine the gender of a voice, although it may not be suitable for tasks that require quick responses.
