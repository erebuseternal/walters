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
