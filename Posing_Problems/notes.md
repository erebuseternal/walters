# Notes

### November 20, 2019
I've been working on posing an RL problem. I'd just like to jot down what I've learned so far. 

The first part to an RL problem is posing the utility function you're trying to optimize. From there you can work back to the things you need in order to predict that utility. Obviously you continue working back from there until you reach data you actually have your hands on. Then you build models to simulate all the way up to your utility, based on the entities within your application. The features in your simulation at this point will comprise a superset of the features you need in your agent state because the features capture everything about the utility. 

Now you need all of this because you need the simulator first. Without the simulator you cannot train the RL agent. 

The next step is to define as simple an MVP as possible, so that you have a proof of concept (great for yourself and for stakeholders) and so that you have something to iterate around. Getting something up and running as fast as possible allows you to fail quickly.

So so far I've found the following helpful. 
1. Read how others have approached similar problems (and how they've posed those problems)
2. Define your utility
3. Next ask what's needed to measure your utility
4. Given you're going to model that, *ask where you can get tagged data* (like in my case speed data) - you'll quickly learn what kind of granularity you have, whether you need to start generating data, what kinds of transformations you'll make, etc. 
5. Work back from there to the hard data
6. Figure out what models are already in place
7. Define the most basic version of your game possible