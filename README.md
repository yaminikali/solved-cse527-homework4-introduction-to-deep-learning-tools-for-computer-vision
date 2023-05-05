Download Link: https://assignmentchef.com/product/solved-cse527-homework4-introduction-to-deep-learning-tools-for-computer-vision
<br>
This project is an introduction to deep learning tools for computer vision. You will design and train deep convolutional networks for scene recognition using <u><a href="http://pytorch.org/">P</a></u><a href="http://pytorch.org/">y</a><u><a href="http://pytorch.org/">Torch </a></u><a href="http://pytorch.org/">(</a><u><a href="http://pytorch.org/">http://p</a></u><a href="http://pytorch.org/">y</a><u><a href="http://pytorch.org/">torch.or</a></u><a href="http://pytorch.org/">g)</a>. You can visualize the structure of <a href="http://vision03.csail.mit.edu/cnn_art/index.html">the network with [mNeuron] (</a><u><a href="http://vision03.csail.mit.edu/cnn_art/index.html">http://vision03.csail.mit.edu/cnn_art/index.html (http://vision03.csail.mit.edu/cnn_art/index.html)</a></u><a href="http://vision03.csail.mit.edu/cnn_art/index.html">)</a>

Remember Homework 3: Scene recognition with bag of words. You worked hard to design a bag of features representations that achieved 60% to 70% accuracy (most likely) on 16-way scene classification. We’re going to attack the same task with deep learning and get higher accuracy. Training from scratch won’t work quite as well as homework 3 due to the insufficient amount of data, fine-tuning an existing network will work much better than homework 3.

In Problem 1 of the project you will train a deep convolutional network from scratch to recognize scenes. The starter codes gives you methods to load data and display them. You will need to define a simple network architecture and add jittering, normalization, and regularization to increase recognition accuracy to 50, 60, or perhaps 70%. Unfortunately, we only have 2,400 training examples so it doesn’t seem possible to train a network from scratch which outperforms hand-crafted features

For Problem 2 you will instead fine-tune a pre-trained deep network to achieve about 85% accuracy on the task.

We will use the pretrained AlexNet network which was not trained to recognize scenes at all.

These two approaches represent the most common approaches to recognition problems in computer vision today — train a deep network from scratch if you have enough data (it’s not always obvious whether or not you do), and if you cannot then instead fine-tune a pre-trained network.

There are 2 problems in this homework with a total of 110 points including 10 bonus points. Be sure to read <strong>Submission Guidelines</strong> below. They are important. For the problems requiring text descriptions, you might want to add a markdown block for that.

<h1>Dataset</h1>

Save the <u><a href="https://drive.google.com/open?id=1NWC3TMsXSWN2TeoYMCjhf2N1b-WRDh-M">dataset</a></u><a href="https://drive.google.com/open?id=1NWC3TMsXSWN2TeoYMCjhf2N1b-WRDh-M">(</a><u><a href="https://drive.google.com/open?id=1NWC3TMsXSWN2TeoYMCjhf2N1b-WRDh-M">click me) </a></u><a href="https://drive.google.com/open?id=1NWC3TMsXSWN2TeoYMCjhf2N1b-WRDh-M">(</a><u><a href="https://drive.google.com/open?id=1NWC3TMsXSWN2TeoYMCjhf2N1b-WRDh-M">https://drive.</a></u><a href="https://drive.google.com/open?id=1NWC3TMsXSWN2TeoYMCjhf2N1b-WRDh-M">g</a><u><a href="https://drive.google.com/open?id=1NWC3TMsXSWN2TeoYMCjhf2N1b-WRDh-M">oo</a></u><a href="https://drive.google.com/open?id=1NWC3TMsXSWN2TeoYMCjhf2N1b-WRDh-M">g</a><u><a href="https://drive.google.com/open?id=1NWC3TMsXSWN2TeoYMCjhf2N1b-WRDh-M">le.com/open?id=1NWC3TMsXSWN2TeoYMC</a></u><a href="https://drive.google.com/open?id=1NWC3TMsXSWN2TeoYMCjhf2N1b-WRDh-M">j</a><u><a href="https://drive.google.com/open?id=1NWC3TMsXSWN2TeoYMCjhf2N1b-WRDh-M">hf2N1b-WRDh-M) </a></u>into your working folder in your Google Drive for this homework.

Under your root folder, there should be a folder named “data” (i.e. XXX/Surname_Givenname_SBUID/data) containing the images. <strong>Do not upload</strong> the data subfolder before submitting on blackboard due to size limit.

There should be only one .ipynb file under your root folder Surname_Givenname_SBUID.

<h1>Some Tutorials (PyTorch)</h1>

<table>

 <tbody>

  <tr>

   <td width="59"></td>

  </tr>

  <tr>

   <td></td>

   <td></td>

  </tr>

 </tbody>

</table>

You will be using PyTorch for deep learning toolbox (follow the <u><a href="http://pytorch.org/">link </a></u><a href="http://pytorch.org/">(</a><u><a href="http://pytorch.org/">http://p</a></u><a href="http://pytorch.org/">y</a><u><a href="http://pytorch.org/">torch.or</a></u><a href="http://pytorch.org/">g)</a> for installation).

<a href="http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html">For PyTorch beginners, please read this </a><u><a href="http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html">tutorial</a></u>

<u><a href="http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html">(http://p</a></u><a href="http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html">y</a><u><a href="http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html">torch.or</a></u><a href="http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html">g</a><u><a href="http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html">/tutorials/be</a></u><a href="http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html">g</a><u><a href="http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html">inner/deep_learnin</a></u><a href="http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html">g</a><u><a href="http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html">_60min_blitz.html)</a></u> before doing your homework.

Feel free to study more tutorials at <u><a href="http://pytorch.org/tutorials/">http://p</a></u><a href="http://pytorch.org/tutorials/">y</a><u><a href="http://pytorch.org/tutorials/">torch.or</a></u><a href="http://pytorch.org/tutorials/">g</a><u><a href="http://pytorch.org/tutorials/">/tutorials/ </a></u><a href="http://pytorch.org/tutorials/">(</a><u><a href="http://pytorch.org/tutorials/">http://p</a></u><a href="http://pytorch.org/tutorials/">y</a><u><a href="http://pytorch.org/tutorials/">torch.or</a></u><a href="http://pytorch.org/tutorials/">g</a><u><a href="http://pytorch.org/tutorials/">/tutorials/)</a></u><a href="http://pytorch.org/tutorials/">.</a>

Find cool visualization here at <u><a href="http://playground.tensorflow.org/">http://play</a></u><a href="http://playground.tensorflow.org/">g</a><u><a href="http://playground.tensorflow.org/">round.tensorflow.or</a></u>

<h1>Starter Code</h1>

In the starter code, you are provided with a function that loads data into minibatches for training and testing in PyTorch.

<h1>Problem 1: Training a Network From Scratch</h1>

{Part 1: 35 points} Gone are the days of hand designed features. Now we have end-to-end learning in which a highly non-linear representation is learned for our data to maximize our objective (in this case, 16-way classification accuracy). Instead of 70% accuracy we can now recognize scenes with… 25% accuracy. OK, that didn’t work at all. Try to boost the accuracy by doing the following:

<strong>Data Augmentation</strong>: We don’t have enough training data, let’s augment the training data. If you left-right flip

(mirror) an image of a scene, it never changes categories. A kitchen doesn’t become a forest when mirrored. This isn’t true in all domains — a “d” becomes a “b” when mirrored, so you can’t “jitter” digit recognition training data in the same way. But we can synthetically increase our amount of training data by left-right mirroring training images during the learning process.

After you implement mirroring, you should notice that your training error doesn’t drop as quickly. That’s actually a good thing, because it means the network isn’t overfitting to the 2,400 original training images as much (because it sees 4,800 training images now, although they’re not as good as 4,800 truly independent samples). Because the training and test errors fall more slowly, you may need more training epochs or you may try modifying the learning rate. You should see a roughly 10% increase in accuracy by adding mirroring. You are <strong>required</strong> to implement mirroring as data augmentation for this part.

You can try more elaborate forms of jittering — zooming in a random amount, rotating a random amount, taking a random crop, etc. These are not required, you might want to try these in the bonus part.

<strong>Data Normalization</strong>: The images aren’t zero-centered. One simple trick which can help a lot is to subtract the mean from every image. It would arguably be more proper to only compute the mean from the training images (since the test/validation images should be strictly held out) but it won’t make much of a difference. After doing this you should see another 15% or so increase in accuracy. This part is <strong>required</strong>.

<strong>Network Regularization</strong>: Add dropout layer. If you train your network (especially for more than the default 30 epochs) you’ll see that the training error can decrease to zero while the val top1 error hovers at 40% to 50%. The network has learned weights which can perfectly recognize the training data, but those weights don’t generalize to held out test data. The best regularization would be more training data but we don’t have that. Instead we will use dropout regularization.

What does dropout regularization do? It randomly turns off network connections at training time to fight overfitting. This prevents a unit in one layer from relying too strongly on a single unit in the previous layer. Dropout regularization can be interpreted as simultaneously training many “thinned” versions of your network. At test, all connections are restored which is analogous to taking an average prediction over all of the “thinned” <a href="https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf">networks. You can see a more complete discussion of dropout regularization in this </a><u><a href="https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf">paper (https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)</a></u><a href="https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf">.</a>

The dropout layer has only one free parameter — the dropout rate — the proportion of connections that are randomly deleted. The default of 0.5 should be fine. Insert a dropout layer between your convolutional layers. In particular, insert it directly before your last convolutional layer. Your test accuracy should increase by another 10%. Your train accuracy should decrease much more slowly. That’s to be expected — you’re making life much harder for the training algorithm by cutting out connections randomly.

If you increase the number of training epochs (and maybe decrease the learning rate) you should be able to achieve around 50% test accuracy. In this part, you are <strong>required</strong> to add dropout layer to your network.

Please give detailed descriptions of your network layout in the following format:

Data augmentation: [descriptions]

Data normalization: [descriptions]

Layer 1: [layer_type]: [Parameters]

Layer 2: [layer_type]: [Parameters]

…

Then report the final accuracy on test set and time consumed for training and testing separately.

{Part 2: 15 points} Try <strong>three techniques</strong> taught in the class to increase the accuracy of your model. Such as increasing training data by randomly rotating training images, adding batch normalization, different activation functions (e.g., sigmoid) and model architecture modification. Note that too many layers can do you no good due to insufficient training data. Clearly describe your method and accuracy increase/decrease for each of the three techniques.

<strong>Description!</strong>

Data augmentation: Each image is flipped at axis 1 using numpy and appended to the dataset, all the labels are appened again in labels as labels will remain same even after flipping images.

Data normalization: Each image array values are subtracted and divided by that particular image’s mean and standard deviation respectively. From experience, such normalization is really helpful.

cannot die whereas RELUs do(vanishing gradient problem). SELU has one more advantage of self normalization that is why I <strong>didn’t use batchNorm layers</strong> here. Results show significant improvements not in terms of accuracy though i.e. <strong>62%</strong>. Comparing 1st experiment with this one, here I don’t see overfitting yet (from the loss-epoch plots) and tweaking few parameters would definitely show improvement in accuracy too for same number of epochs. This network was trained on <strong>random augmented dataset same as used in experiment 1</strong>. I also tried sigmoid activation function which did not perform well compared to RELU. I believe reasons would be sigmoid’s saturation and killing gradients, plus its outputs are not zero centered whereas RELU has its own advantages too (other than RELU is good at what sigmoid is bad at) which are computationally efficient and faster convergence rate.

Problem 2: Fine Tuning a Pre-Trained Deep Network

{Part 1: 30 points} Our convolutional network to this point isn’t “deep”. Fortunately, the representations learned by deep convolutional networks is that they generalize surprisingly well to other recognition tasks.

<a href="http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks">But how do we use an existing deep network for a new recognition task? Take for instance, </a><u><a href="http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks">AlexNet</a></u>

<u><a href="http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks">(http://papers.nips.cc/paper/4824-ima</a></u><a href="http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks">g</a><u><a href="http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks">enet-classification-with-deep-convolutional-neural-networks)</a></u> network has 1000 units in the final layer corresponding to 1000 ImageNet categories.

<strong>Strategy A</strong>: One could use those 1000 activations as a feature in place of a hand crafted feature such as a bagof-features representation. You would train a classifier (typically a linear SVM) in that 1000 dimensional feature space. However, those activations are clearly very object specific and may not generalize well to new recognition tasks. It is generally better to use the activations in slightly earlier layers of the network, e.g. the 4096 activations in the last 2nd fully-connected layer. You can often get away with sub-sampling those 4096 activations considerably, e.g. taking only the first 200 activations.

<strong>Strategy B</strong>: <em>Fine-tune</em> an existing network. In this scenario you take an existing network, replace the final layer (or more) with random weights, and train the entire network again with images and ground truth labels for your recognition task. You are effectively treating the pre-trained deep network as a better initialization than the random weights used when training from scratch. When you don’t have enough training data to train a complex network from scratch (e.g. with the 16 classes) this is an attractive option. Fine-tuning can work far better than

<a href="http://www.cc.gatech.edu/~hays/papers/deep_geo.pdf">Strategy A of taking the activations directly from an pre-trained CNN. For example, in </a><u><a href="http://www.cc.gatech.edu/~hays/papers/deep_geo.pdf">this paper</a></u>

<u><a href="http://www.cc.gatech.edu/~hays/papers/deep_geo.pdf">(http://www.cc.</a></u><a href="http://www.cc.gatech.edu/~hays/papers/deep_geo.pdf">g</a><u><a href="http://www.cc.gatech.edu/~hays/papers/deep_geo.pdf">atech.edu/~ha</a></u><a href="http://www.cc.gatech.edu/~hays/papers/deep_geo.pdf">y</a><u><a href="http://www.cc.gatech.edu/~hays/papers/deep_geo.pdf">s/papers/deep_</a></u><a href="http://www.cc.gatech.edu/~hays/papers/deep_geo.pdf">g</a><u><a href="http://www.cc.gatech.edu/~hays/papers/deep_geo.pdf">eo.pdf)</a></u><a href="http://www.cc.gatech.edu/~hays/papers/deep_geo.pdf"> from CVPR 2015, there wasn’t enough d</a>ata to train a deep network from scratch, but fine tuning led to 4 times higher accuracy than using off-the-shelf networks directly.

You are required to implement <strong>Strategy B</strong> to fine-tune a pre-trained <strong>AlexNet</strong> for this scene classification task. You should be able to achieve performance of 85% approximately. It takes roughly 35~40 minutes to train 20 epoches with AlexNet.

Please provide detailed descriptions of:

<ul>

 <li>which layers of AlexNet have been replaced</li>

 <li>the architecture of the new layers added including activation methods (same as problem 1)</li>

 <li>the final accuracy on test set along with time consumption for both training and testing</li>

</ul>

{Part 2: 20 points} Implement Strategy A where you use the activations of the pre-trained network as features to train one-vs-all SVMs for your scene classification task. Report the final accuracy on test set along with time consumption for both training and testing.

<a href="https://pytorch.org/docs/stable/_modules/torchvision/models/vgg.html">{Bonus: 10 points} Bonus will be given to those who fine-tune the </a><u><a href="https://pytorch.org/docs/stable/_modules/torchvision/models/vgg.html">VGG network</a></u>

<u><a href="https://arxiv.org/pdf/1409.1556.pdf">(https://p</a></u><a href="https://arxiv.org/pdf/1409.1556.pdf">y</a><u><a href="https://arxiv.org/pdf/1409.1556.pdf">torch.or</a></u><a href="https://arxiv.org/pdf/1409.1556.pdf">g</a><u><a href="https://arxiv.org/pdf/1409.1556.pdf">/docs/stable/_modules/torchvision/models/v</a></u><a href="https://arxiv.org/pdf/1409.1556.pdf">gg</a><u><a href="https://arxiv.org/pdf/1409.1556.pdf">.html)</a></u> <u><a href="https://arxiv.org/pdf/1409.1556.pdf">paper</a></u>

<u><a href="https://arxiv.org/pdf/1409.1556.pdf">(https://arxiv.or</a></u><a href="https://arxiv.org/pdf/1409.1556.pdf">g</a><u><a href="https://arxiv.org/pdf/1409.1556.pdf">/pdf/1409.1556.pdf)</a></u><a href="https://arxiv.org/pdf/1409.1556.pdf"> and compare performance with AlexNet. </a>Explain why VGG performed better or worse.

<strong>Hints</strong>:

<a href="http://pytorch.org/docs/master/torchvision/models.html">Many pre-trained models are available in PyTorch at </a><u><a href="http://pytorch.org/docs/master/torchvision/models.html">here (http://p</a></u><a href="http://pytorch.org/docs/master/torchvision/models.html">y</a><u><a href="http://pytorch.org/docs/master/torchvision/models.html">torch.or</a></u><a href="http://pytorch.org/docs/master/torchvision/models.html">g</a><u><a href="http://pytorch.org/docs/master/torchvision/models.html">/docs/master/torchvision/models.html)</a></u><a href="http://pytorch.org/docs/master/torchvision/models.html">.</a>

For fine tuning pretrained network using PyTorch please read this tutorial


