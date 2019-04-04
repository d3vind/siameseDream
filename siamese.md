# <center>Siamese Dream</center>
## <center>A Beginners Guide to Tracking with Siamese Neural Networks</center>

You're sitting on your couch, dog at your feet, one hand completely immersed in a bag of white cheddar Smartfood. Netflix is on the box and you're just passing minute 48 of the imdb rated 6.4/10, thriller Surveillance. You watch as the main character tries to evade the watching eye of the police. As they pore over endless frames of CCTV following every move of the antagonist without touchig a button. You cant help but think. They can't actually do this, can they? The answer to that question is yes, _they_ absolutely can. One way they could do this is using Siamese neural networks.
Now let this be warning, this article isn't going to make Elon Musk and you may not come out of with a fully functioning prototype but by the end you will understand exactly how these networks are able to achieve such spooky results.
Assumed is a basic knowledge of convolution neural networks and the math behind them. If you don't have that I recommend saving this tab and watching this video, then coming back.
What we need to know

[Learn about Convolutional Networks][0f7dd8c6]

[0f7dd8c6]: https://www.youtube.com/watch?v=YRhxdVk_sIs "link"


<center>![snowmen](blogPhotos/snow-men.jpg)
## Siamese Overview

As the name suggests the basis of this network is indeed twinned identical fully convolution networks _(we'll get to that_). The twins goal is to create the same representation of the most important features of two distinct images and discern the target from background.
<center>![Alt Text](https://media.giphy.com/media/YTK49xvHgteJuMC0pn/giphy.gif)

Within the application of video tracking these two images come from two temporally distinct places(ie. different frames) with a only a small distance between them. The basic idea is that if we can take the representation of our first image and correlate it with our 2nd the area with this high correlation will be the new location of the target we want to track. We now take this image and compare it to the next frame and do so until the end of the video. Ok that was a lot of words, if it seems confusing reference the photo below. Now pat yourself on the back, you just learned the absolute basics of how this works. If you're studying this for an exam or to impress your friends this is probably a good place to stop reading. If you're looking for the nitty gritty details stay tuned.

As mentioned above the main focus of the this work is the training of the siamese network in order to learn a similarity network between patches within the same video

So let's tackle this network. At the crux of why this is such an effective way to track video in real time are three concepts that we need to go over. These are transfer learning, fully convolutional networks and discriminant correlation filers.
So lets We already know what a CNN so lets look at what makes one fully convolutional.

### Fully Convolution Networks
Fully convolutional indicates that the neural network is composed of convolutional layers without any fully-connected layers at the end of the network.

<center>![conv image](https://i.imgur.com/LiFHufm.png)

The main difference with that of a fully convolutional net and a traditional is that our fully convolutional nework is
 learning filters every where. As mentioned above it does not contain a fully connected layer instead  the decision-making layers at the end of the network are filters.

    Why do we want to do this?
    Input image size:  If you don’t have any fully connected layer in your network, you can apply the network to images of virtually any size. Because only the fully connected layer expects inputs of a certain size, which is why in architectures like AlexNet, you must provide input images of a certain size (224x224). In our case this allows us to easily adjust to meet the specifications necessary to process any video. As there is no one size fits all in regards to digital media this makes the implementation more robust.

    __**Computational cost and representation power**__ : There is also a distinction in terms of compute vs storage between convolutional layers and fully connected layers.
    Do to the nature of being fully connected it eats up far more computation. For this reason researchers are tending towards fully conovolutional networks and in our case it doesn't hurt as we are operating in real time.


### Transfer Learning
Very generally transfer learning is used when an network does not start learning from scratch. Rather it uses already trained knowledge and builds on top of it. An example I can think of is a mechanic, he may have worked for years at volkswagan dealership fixing cars, perhaps he gets a new job fixing school buses. He already knows how to fix vehciles and knows the intricacies of doing so. Although he needs to specialize his skills and learn more to transfer this knowledge to a new type of vehicle. Transfer learning in AI works on the same premise. In our case the first couple layers are pretrained and hold the weights of notable network "alexNet" they will offer to give a general representation of the image as outlined above. The remainder of the layers are trained on large datasets such as [GOT-10K](http://got-10k.aitestunion.com/) with predefined bounding boxes and targets. The loss of the function represented by the error of the network(the predicted location in comparison to the ground truth)

      Why do we want to do this?
        1) Speeds up network and training
        2) Helps prevent overfitting





The peak of the correlation map is supposed to be located at the center of the map (because the images are both centered in the target).



### Correlation Filters
Ok, you're smart, you probably already know this but we'll go over it just in case. Remember the operation a convolution neural network performs? How it takes the the filter and performs this matrix math within a sliding box over the whole image to produce another different representation of the same image. This is almost identically the same for correlation filters but instead of transferring one image your making a representation of two images and finding a way to represent their similarities.
This sounds like a lotttttt of stuff to happen in real time. Thankfully we are saved by the power of mathematics again. These happens in the fourier domain. In regular people speak that means that your computer can compute the operations of these sliding windows to be processed in parallel by the cores on your GPU. This allows you to observe a result almost instantaneously.

You might be thinking, how can this be happening in real time this is a lot of mathematics to be happening to process this in real time.
How do they do this? Welllllllllll
<center>![Alt Text](https://media.giphy.com/media/YkgoW1fPJr4ovVUGGS/giphy.gif
)

    Why do we want to do this?
      1) Correlation filters are fast and can work in real time
      2) Effective

Lets throw this all together. Take a look at the image below, if you understand it, it's time to take over the world.

<center>![](https://i.imgur.com/HSIyc98.jpg)



### What Can we do with this in the Future
Building on this there is a lot of potential applications that start with continuing to improve the current models to be more robust and handle problems in occlusion
The tecnhnique described in the this article is based heavily on the ternary paper on the subject siamFC: by Luca Bernitto
I recommend further readings in more complex approaches in the same idea siamRPN, SiamMask ......List LINKS
