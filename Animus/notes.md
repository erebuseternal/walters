# Notes

### January 7, 2020

#### General Notes
Definitely need to get this data standardized and formatted in a reasonable way. Couple of things I need to do:
- Remove the timestamp and other data from the top and bottom of these frames. I don't want to give my algorithm the chance to overfit to that cruft
- Reduce the resolution - these images are quite large. I want to reduce their size to help with training
- Pad the images to get them all to the same size

Basically I'm going to start by building a data transformer to take in an image and transform it to the right resolution, shape, and get rid of the cruft. Then I'll take all my images and run them through this processor. 

To the end of getting rid of the cruft there seem to be several different trap cameras. Some examples:

3.jpg - black border (Reconyx)
6023.jpg - black border (stag in left lower corner)
6125.jpg - white border (bushnell orange)
6557.jpg - white border (bushnell boring)
6579.jpg - white border(ltl acorn)

Margin seems reasonable consistent... I think we can just crop and pad tbh. Just need to figure out by how much.

Here's a printout of the sizes:

{(1080, 1920, 3), (3000, 4000, 3), (1944, 2592, 3), (1520, 2736, 3), (1836, 3264, 3), (480, 640, 3), (1456, 2592, 3), (2448, 4352, 3), (2448, 3264, 3), (2550, 3400, 3), (2160, 3840, 3), (2480, 4416, 3), (768, 1024, 3), (1832, 3264, 3), (1800, 2400, 3), (1920, 2560, 3), (1512, 2688, 3), (2080, 3744, 3), (1536, 2048, 3)}

Okay so I can crop by using the following algorithm:

1. Determine (for the right or left edge) the bottom and top pixel values
2. Find the first index where the pixel changes value
3. Use those values to crop out these borders

Then I can simply pad to get the same aspect ratio, resize to get the correct number of pixels, and then input the data. 

--> Took a really long time to loop through 

#### Accomplished Today
- Found data sizes
- Investigated the borders on these images
- Learned about opencv
- Determined a method for doing cropping, resizing, and general standardization
#####2hrs

### January 9, 2020

#### General Notes
Let's try building that converter today. :)

Okay looking at the y/x ratios above it seems the largest is 1.8. So I'm thinking we go for wider rather than squarer images as the base. That will mean we add some padding to stretch out the squatter images. 

Looking at 300x540 it seems we could still go smaller. 150x270 seems alright. Going smaller really washes out any kind of detail. These images are already many orders of magnitude smaller than they originally were. (I will check out some papers though to see what people have done.)

Goodness... I didn't even spot the animals in some of these when it's down at such a low resolution... I really need to look at what people are using. I'm going to convert down to 340 by 540 in order to make these more portable though.

Found this paper: https://www.pnas.org/content/pnas/115/25/E5716.full.pdf. Gotta go get ink for my printer though. :) 

#### Accomplished Today
- Built standardizer
- Standardized data set
- Found first paper
- Got ink :P
#####2.5hrs