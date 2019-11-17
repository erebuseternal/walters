# Notes

## Building Environments

### November 13, 2019
I just built the gridworld environment and it took me way too long. Ended up rewriting it over and over again because I didn't take any time to design the class. Started with numpy arrays for state, then switched to tuples, then switched to instance parameters. Each time I had to refactor a whole bunch of methods. I also kept rewriting which methods I needed over and over again. Next time I should definitely design what I'm going to do and what the data structures will be first.

#### Goals for Next Time
- Build the time difference method (the other methods seem totally inapplicable to any real world problems and I need to get onto the methods we're actually going to use)

### November 14, 2019
Rebuilt the gridworld and it's so much better now. Really important bits were to allow state indexing, to create an action and rewards map, and to return the reward on the action. So easy to use now and allowed me to put together the monte carlo stuff quickly and with very few bugs. Which brings me to the point that I have policy evaluation done for monte carlo! So now I just need to do the policy optimization step. This is a lot of fun when done right :) 

#### Goals for Next Time
- Build the Monte Carlo policy optimization
- Try out step penalized grid world

Also, just as a side note, I should do the coding as I have the lectures. Otherwise I just end up forgetting and having to watching the lectures over again (like I'm having to do now...) So just keep a steady pace, and it's really a lot of fun and not frustrating, so no need worrying about getting into a "fugue" state. :)

### November 15, 2019
In building the TD(0) model I realized that you really have to allow them to explore _a lot_. If you don't they'll create cycles and quickly learn those cycles over everything else which gets incredibly annoying. So just a note to remember. 

In building the approximation model I found my parameters were diverging. Only closer inspection it became clear that they were pinging back and forth between negative and positive at larger and larger intervals. Decreasing the larning rate (alpha) fixed this right up! :)

#### Accomplished Today
- Built Monte Carlo and TD(0) methods
- Tried them on basic and costly worlds
- Built TD(0) Function Approximation method
#####4hrs

#### Goals for Next Time
- See how far you can get on the final project (tomorrow is a 5 hour day)

### November 16, 2019
In building the investor I found that my model was diverging like crazy again. Turned out this was simply because I wasn't normalizing. So make sure you normalize!

#### Accomplished Today
- Built the investor (final project)
- Finished up the RL intro class!
#####6hrs

#### Goals for Next Time
- Start DNN2 class
