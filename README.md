# Description

There are two approaches for deriving Reinforcement learning algorithms,



![two](https://user-images.githubusercontent.com/19307995/42143368-0003521a-7db5-11e8-8b26-4476fcf7347d.png)

As the image illustrates, policy optimization just looks at the problem as an optimization problem, where you're trying to optimize your expected reward and have some parameters of your policy. The derivative free optimization (DFO) or evolutionary methods ignore the structure of the problem, and say we can take a parameter and get a noisy estimate of how good is it and then try to move to the part of the parameters space where we're getting better performance.

The objective here is to maximize E [R | π(.,θ)]. The DFO just looks at the problem as a black box where we put in θ and then something complicated happens and then comes the rewards of the episode. Cross entropy is an evolutionary algorithm because at every point in time it's maintaining a distribution over the policy parameter vector. It's like if you have a population of individuals and some of them have higher fitness than others, so your distribution moves towards the individuals with higher fitness. 
