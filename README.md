
# Active screening

Current approach in drug screening is to test entire chemical library space on a given target to find an active compounds. However, the brute force approach takes lot of time, money, and do not learn in picking good compounds from prior experiments. To overcome the problem, we can let machine learn the features of a small molecules that responded to the target for subset of the experiments. Then we can ask the machine to provide the next best small molecule pick that responds to the target. This approach will drastically limit the time and resources to serarch chemical space.

Here I have implemented the active learning routine that picks small molecule compounds with high chance of responding to the target. The dataset used for the assignment can be found here - http://www.biostat.wisc.edu/~page/Thrombin.zip

Four different strategies were used - Random, Max probability, Max Entropy, and Optimal.

- Optimal classifier: artificially awlays picks hits until none left
- Random classifier: picks compounds completely at random, simulating the high throughput screening
- Max entropy: picks compounds based on max entropy
- Max probability: picks compounds based on max probability of being active

Since the input data is big, i have created a pickle object that can be found here -

Use the pickle object as input.

## Usage

`python active_learning_drug_discovery.py drug_discovery_data`

## Returns
Plot showing different strategies %(hits discovered) vs %(compounds tested)

![alt text](https://github.com/virenar/active_screening/blob/master/performance_active_learner_comparision.png)
