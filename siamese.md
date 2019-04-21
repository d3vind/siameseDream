# <center>Siamese Dream</center>
## <center>A Beginners Guide to Tracking with Siamese Neural Networks</center>

You're sitting on your couch, dog at your feet, one hand completely immersed in a bag of White Cheddar Smartfood. Netflix is on the TV and you're just passing minute 48 of the IMDB rated 6.4/10, thriller Surveillance. You watch as the main character tries to evade the watching eye of the police. Poring over endless frames of CCTV surveillance footage, they follow every move of the antagonist without touching a single button. You cant help but think. They can't actually do this, can they?  The answer to that question is yes, it is absolutely possible. This example displays one of the many possible applications within the popular scientific discipline of computer vision. A growing area area of research, that has seen an explosion of use cases and promising algorithms such as those involving a possible solution to our answer above. Siamese Neural Networks.

Now let this be warning, this article will not make you an over night Elon Musk and you may not come out of with a fully functioning prototype like other articles on this website. What you will have, is an understanding of what these networks are and in detail how they achieve such results.

Assumed is a basic knowledge of convolution neural networks and the math behind them. If you don't have that I recommend saving this tab and watching this video, then coming back.
What we need to know

[Learn about Convolutional Networks][0f7dd8c6]

[0f7dd8c6]: https://www.youtube.com/watch?v=YRhxdVk_sIs "link"


<center>![snowmen](blogPhotos/snow-men.jpg)
## Siamese Overview

Let's consider our end goal as taking a video, selecting a target and tracking it throughout the duration. Now lets peer down the rabbit hold and abstract this. First lets think about exactly what video is. Fundamentally it is just a series of photos or images displayed in rapid succession. So all we are trying to do is find a find a target in every frame of a video and draw a box around it. This sounds eerily similar to everyone's favourite children's book _"Where's Waldo"_.
<center>![Alt Text](https://media.giphy.com/media/d83PLQh2whV1inCSZ6/giphy.gif)

Using our eye's and maybe our finger,  we scan the page from left to right looking at every face, shirt or feature to see if it matches that of our old pal Waldo. Maybe even drawing a circle around him just to ruin the game for anyone else that might want to try after. As we can't do this for 200 frames a second. This is exactly what we want to train our network to do. As said above our network will look at a photo and draw a box around the choose target.

<center>![Alt Text](https://media.giphy.com/media/j2Se3clmPC0S8q11E7/giphy.gif)

Unfortunately our network has a bad memory for images so we must constantly remind it what the target image looks like. For this reason we actually need two networks. This is where the whole simaese thing comes in. Just like siamese twins. We are have two different but similiar images that we are going to treat exactly the same.  We need to process our two images(target and entire frame) and find their features in the exact same way. We do this with twinned identical fully convolution networks _(we'll get to that_). We convolve both images in identically structured networks trained for to best differentiate the most useful features. The twins goal is to create the same representation of the most important features (maybe waldos shirt?) of two distinct images and discern the target from background.

<center>![Alt Text](https://media.giphy.com/media/YTK49xvHgteJuMC0pn/giphy.gif)

The basic idea is that if we can take the representation of our first image and place it on top of our second we can find where they are most similiar(highest correlation) and this must be the location of our target image in the new photo. We now just repeat this for all the frames until the end of the video! Pat yourself on the back, you just learned the absolute basics of how this works. If you're studying this for an exam or to impress your friends this is probably a good place to stop reading. If you're looking for the nitty gritty details stay tuned.

At the crux of why this is such an effective way to track video in real time are three concepts that we need to go over. These are transfer learning, fully convolutional networks and discriminant correlation filers.
We already know what a convolutional neural network is so lets look at what makes one fully convolutional.

## Fully Convolution Networks
Fully convolutional indicates that the neural network is composed of convolutional layers without any fully-connected layers at the end of the network.

<center>![conv image](https://i.imgur.com/LiFHufm.png)

As mentioned above the main difference with that of a fully convolutional net and a traditional is that our fully convolutional network has fully connected layers. This means it is learning filters everywhere, including this final decision making layer.
#### Why do we want to do this?
_**Input image size:**_  If you don’t have any fully connected layer in your network, you can apply the network to images of virtually any size. Because only the fully connected layer expects inputs of a certain size, which is why in architectures like [AlexNet][772ec5d6]
. You must provide input images of a certain size. As most videos and images have varying sizes,this allows us to easily adjust to meet the specifications necessary to process elastically. In this case using a fully convolutional network makes our implementation more robust.


  [772ec5d6]: https://medium.com/@smallfishbigsea/a-walk-through-of-alexnet-6cbd137a5637 "a"

  _**Computational cost and representation power**_ : There is also a distinction in terms of compute vs storage between convolutional layers and fully connected layers.
    Do to the nature of being fully connected it eats up far more computation. In a fully connected layer each neuron is connected to every neuron in the previous layer. Each connection has it's own weight. In a convolutional layer each neuron is only connected to a few nearby local neurons in the previous layer, and the same set of weights is used for every neuron. This works perfectly for looking at images where, as required, the features are local (e.g. "waldo's red and white shirt" consists of a set of nearby pixels, not spread all across the image). Having fewer connections makes the fully convolutional network faster and as we are operating in real time this is crucial to effectiveness.



## Transfer Learning
Very generally transfer learning is used when an network does not start learning from scratch. Rather it uses already trained knowledge and builds on top of it. An example I can think of is a mechanic, he may have worked for years at Volkswagen dealership fixing cars, perhaps he gets a new job fixing school buses. He already knows how to fix vehicles and knows the intricacies of doing so. Although he needs to specialize his skills and learn more to transfer this knowledge to a new type of vehicle. Transfer learning in machine learning works on the same premise. In our case the first couple layers are pretrained and hold the weights of a large notably successful network [AlexNet](https://mediuAlexNet"m.com/@smallfishbigsea/a-walk-through-of-alexnet-6cbd137a5637). This will offer to give a general representation of the image as outlined above. The remainder of the layers are trained on large datasets such as [GOT-10K](http://got-10k.aitestunion.com/) with predefined bounding boxes and targets. The loss of the function represented by the error of the network(the predicted location in comparison to the ground truth)

      Why do we want to do this?
        -It speeds up network and training!







## Correlation Filters
Alright we are really starting to put the pieces together now. Let's think back to our Where's Waldo analogy. We currently have a a representation of what we are looking for in waldo (glasses, shirt hat). All we have to do is compare this to our frame features and see if we can match them up!
 If you did your homework you remember the operation a convolution neural network performs. How it takes the the filter and performs matrix math with a sliding window over the whole image . This produces a new image with accentuated features. In searching for our target we are going to do almost the same thing. Instead of using a filter we will use our search image! Passing it over the image block by block, we convolve the image where the output represents the similarity between the features. The area of our search image that has the highest similarity is where Waldo is!
 I know what you're thinking that sounds like a lot of stuff to occur in real time. Thankfully we are saved by the power of mathetmatics. The reason we can do this is because these operations happen in the [fourier domain](https://betterexplained.com/articles/an-interactive-guide-to-the-fourier-transform/). We don't really need to get into that, just know that it allows your computer to computer the operations of all these sliding windows simultaneously in prallel by the cores of your GPU.This allows us to observe the results almost instantaneously.



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
