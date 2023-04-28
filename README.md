<h3>Neural Network for Malicious HTTP Requests Detection</h3>

<hr>
<h4>Approach</h4><br>
Feedforward Neural Network utilising supervised learning.<br>
Chosen activation function is ReLU (Rectified Linear Unit).<br>
Mini-Batch gradient descent with Adam optimiser used to improve performance and accuracy.<br>

<hr>

<h4>Results</h4>
Trained and tested on CSIC 2010<br>
<b>AVERAGE RESULTS</b> 
(calculated on 2-fold test, 10x retrained and retested on both folds to get average results)<br>
<br>
Overall accuracy	92.165 % <br>
False negatives	3.350 % <br>
False positives	4.485 % <br>

<hr>

<h4>How to run</h4>
Everything is prepared in the main class, program is ready to run. It will conduct training and test on both folds 1 time and show accuracy. During training, you will be able to see calculated error every 10 epochs. <br>
To change parameters of the neural network, navigate into main class, all parameters can be found there, in the main method. <br>
Install NumPy <br>
Navigate into the project directory <br>
Run with command: python3 main.py <br>
<br>
Please remember, the accuracy will be slightly different from the accuracy shown above, since the accuracy above is average calculated on multiple iterations. <br>
