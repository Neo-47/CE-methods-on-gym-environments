# Description

There are two approaches for deriving Reinforcement learning algorithms,

<img src = "https://user-images.githubusercontent.com/19307995/42143368-0003521a-7db5-11e8-8b26-4476fcf7347d.png" width = "1000" margin = "1000"/>

Policy optimization just looks at the problem as an optimization problem, where you're trying to optimize your expected reward and have some parameters of your policy. The derivative free optimization (DFO) or evolutionary methods ignore the structure of the problem, and say we can take a parameter and get a noisy estimate of how good it is and then try to move to the part of the parameters space where we're getting better performance.

The objective here is to maximize E [R | π(.,θ)]. The DFO just looks at the problem as a black box where we put in θ and then something complicated happens and then come the rewards of the episode. Cross entropy is an evolutionary algorithm because at every point in time it's maintaining a distribution over the policy parameter vector. It's like if you have a population of individuals and some of them have higher fitness than others, so your distribution moves towards the individuals with higher fitness. 

If the space of policies is sufficiently small, or can be structured so that good policies are common or easy to find, or if a lot of time is available for the search, then evolutionary methods can be effective. In addition, evolutionary methods have advantages on problems in which  the learning agent cannot sense the complete state of its environment.

However, methods able to take advantage of the details of the individual behavioral interaction can be much more efficient than evolutionary methods in many cases. Evolutionary methods ignore much of the useful structure of the reinforcement learning problem: they do not use the fact that the policy they are searching for is a function from state to actions; they do not notice which states an individual passes through during its lifetime, or which actions it selects. In some cases this information can be misleading (e.g., when states are misperceived), but more often it should enable more efficient search. 



## Continuous State Space

+ CartPole: A reward of +1 is provided for every time step that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.

![ezgif com-video-to-gif](https://user-images.githubusercontent.com/19307995/42143893-ec24217c-7db7-11e8-9543-4891e1d5de7d.gif)


+ LunarLander: If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main engine is -0.3 points each frame. Solved is 200 points. Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt. Four discrete actions available: do nothing, fire left orientation engine, fire main engine, fire right orientation engine. I trained the agent until I got at least +50 average rewards. You can train it longer to get the optimal behavior.

![ezgif com-video-to-gif 1](https://user-images.githubusercontent.com/19307995/42178987-a83df1ce-7e32-11e8-8bc8-1ee48eb1e57d.gif)



## Discrete State Space

+ Taxi: There are 4 locations (labeled by different letters) and your job is to pick up the passenger at one location and drop him off in another. You receive +20 points for a successful dropoff, and lose 1 point for every time step it takes. There is also a 10 point penalty for illegal pick-up and drop-off actions. You can run the policy to see how the agent behaves.

![screenshot from 2018-07-04 00-53-32](https://user-images.githubusercontent.com/19307995/42248342-539ffe7a-7f25-11e8-842d-823bb129edb9.png)



