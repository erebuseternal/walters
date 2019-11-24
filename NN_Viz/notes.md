# Notes

### November 23, 2019
Built a neural net from scratch. Couple of things for the future:
- A single negative sign that was supposed to be positive (and was written as negative in the lecture I was referring to) kept me from getting past a bug for about an hour. Super annoying, but just be careful in the future.
- Hyper parameters are a bitch... was getting nothing from this, bumped up the learning parameter like crazy and boom! we got the separation I was looking for.

### November 25, 2019
After looking at the decision boundaries for the intermediary nodes and reflecting on how NN's work I've realized that all NNs are doing is creating a long slew of conditional logic. For example if we take the example I ran of two concentric circles and 3 hidden nodes all that's happening is the following. First each node in the hidden layer creates an if/else by drawing a line through the original 2d space. This creates a new 3d space with a series of groups each representing the results of the if/else statements. Then the final node draws it's own line thereby creating a final if else statement. To think about this visually, each if else creates a new shape. That shape is then joined with other shapes (or divided in itself) by subsequent if else statements. 

This means that the whole interpretation that NN's are finding noses and the like is really false. It's finding the rough shape of several noses, but not finding the "idea" of a nose. For example if a NN sees a side profile that will be something totally different to it (a different agglomeration of shapes) than a face facing directly on. A face far away will be represented by different shapes than a face close up. In essence the NN seems to be iterating the many possibilities and then seeing if what we have belongs to any of them. Like I said, conditional logic. The only difference is it's made smooth by the sigmoid, tanh's, and relu's thereby allowing you to train the conditional logic using gradient descent. If decision trees could recombine and split again, they would be no different from NN's. 

This of course is deeply unsatisfying. No wonder NN's require so much data and such large sizes. The work because we live in a world with enormous amounts of data and compute power... 
