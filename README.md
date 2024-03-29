# A Simulation of Lettau, M. (1997)

A simulation code in Python to reproduce Lettau Martin (1997)'s model of a portfolio decision model for boundedly rational agents in a financial market.

The original paper:

[Lettau, M. (1997). Explaining the facts with adaptive agents: The case of mutual fund flows. Journal of Economic Dynamics and Control, 21(7), 1117-1147.](https://www.sciencedirect.com/science/article/pii/S0165188997000468)

This paper studies portfolio decisions of boundedly rational agents in a financial market. Learning is modeled via a genetic algorithm (GA). Learning as modeled in this paper leads agents to hold too much risk as compared to the optimal portfolio of rational investors. Moreover, adaptive agents exhibit an asymmetric response after positive and negative returns where the portfolio adjustment is more pronounced after negative returns. It is demonstrated that investors in mutual funds show the same investment patterns as the adaptive agents in the model. A model with entry and exit of agents is able to match the mutual fund data closely.

## The idea

This paper investigates the decision-making of a bounded-rational agent in portfolio purchase. In this case, it is the number of risky assets one should hold. In theory, if the information is symmetric, a rational agent can easily solve the maximization problem and derive the optimal solution. However, in the real world, this is often impossible. Furthermore, since mutual fund investors are the least informed, the adaptive learning model can be more realistic than the rational model. An agent learns from observed outcomes of their investment decision and adjusts their portfolio composition accordingly.

One illustrious learning method is the Genetic Algorithm (GA), inspired by the evolution process observed in nature. Basically, GA is a way the genes (of a species) can transform themselves through many generations in order to survive and adapt to the new environment. In each generation, GA performs the following 3 operations.

First, GA defines fitness criteria to select the best-performing individuals. At the second step, GA performs a CROSSOVER operation. This is also called "mating" because the best individuals are paired randomly to mate with each other. After mating, offspring are born and inherit a mixture of genetic codes from their parents. For instance, imagine the genetic code of one individual can be represented by a binary string. If person 1 "101101" is to be mated with person 2 "011000", a crossover at point (3) slides each string in half and produces 2 new strings, "101000" and "011101". This process makes sure strong blocks of genetic code prevail in the long run. For the third operation, GA performs MUTATION. In our binary example, it is simply to flip 0 to 1 (and vice versa) with a given mutation rate. Mutation introduces new genetic materials, provides variety, and thus avoids the situation where evolution is stuck.

The GA framework is particularly interesting when being applied to the financial market. For the first operation, selection of the best fitted, one can think of criteria such as profit, or utility. An agent would want to hold the assets that generate a good payoff and discard the bad performing ones. For the second operation, the crossover is similar to trading. Two agents can trade the stocks from each other portfolios to see how they will perform. This action exploits rational belief. If 1 agent knows that the other has selected the best portfolio for him, the transaction is more likely to happen and acts as a diversification method. The mutation operation resembles an adventurous investment. The agent adjusts a small part of his portfolio component to see how it performs. This is also known as exploration. In reality, we have seen cases where conservative investors suddenly tried investing in new assets, such as Game Stop or Dogecoin. In a sense, it is not so different from mutation. Thus, adaptive learning with GA is a good simulation of the fund flows, as the paper proposed.

## Simulations in the paper

![average of parameter](https://i.imgur.com/Eq4Ll5B.png)

![variance](https://i.imgur.com/opNfobx.png)

## Our replication

![](output.png)

The simulation shows that although the algorithm is approaching the optimal solution, the average number lingers a little above the optimal α∗ in the benchmark model, which is indicated as the red line. The intuition is that when information is not enough, then agents tend to hold more assets than optimally needed. The author further argues that a piece of information that is crucial to the adaptive learning agent here is observations S. The bias to hold more risky assets tends to increase if the agent has to make a portfolio decision after observing its performance for only a short time. I confirm such a phenomenon by the following simulation.

By randomness, if an agent observes more positive than negative events, he tends to hold more risky assets. And since other agents are doing the same, this creates a bubble where risky assets are favored and trade exclusively. The bias is reduced if an agent is more patient to observe assets realization more times, before updating the portfolio decision.

The results are interesting for 2 reasons. First, it shows that an adaptive learning agent (GA) may never reach the optimal value if the observation of assets realization is not enough. This can explain partially the build-up of a bubble. Agents are not rational enough and information is not sufficient so they take rare events (high returns for risky assets) not so correctly, compared to the rational agents. A bubble might form if all agents behave in such a manner. Second, the framework of GA in the natural evolution process has been nicely introduced and applied in the social world, which in this case is the financial decision. An intuitive GA implementation here shows that a GA model is actually more accessible and realistic than many rational agent models. From this initial work, we can think of many expansions to simulate more complicated models, such as speculative bubbles, production decisions, or propagation/spillover effects in network models.