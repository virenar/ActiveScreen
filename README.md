
# Active screening

The current approach in drug screening is to find active compounds by testing an entire chemical library space on a given target. However, the brute force approach takes a lot of time, resources, and does not learn from prior experiments. To overcome the problem, we can let machine learn the features of a small molecule from a subset of prior experiments that responded to the target. Then we can ask the machine to provide the next best small molecule that responds to the target. This approach drastically reduces the time and resources to search an active compound from an entire chemical space.

Here I have implemented the active learning routine that picks small molecule compounds with a high chance of responding to the target. The dataset used for the assignment can be found here - http://www.biostat.wisc.edu/~page/Thrombin.zip

Four different strategies were used - Random, Max probability, Max Entropy, and Optimal.

- Optimal classifier: artificially always picks hits until none left
- Random classifier: picks compounds completely at random, simulating the high throughput screening
- Max entropy: picks compounds based on max entropy
- Max probability: picks compounds based on max probability of being active

Since the input data is big, i have created a pickle object that can be found here -
https://console.cloud.google.com/m/cloudstorage/b/tinybcket4388/o/drug_discovery_data
Use the pickle object as input.

## Usage

`python active_learning_drug_discovery.py drug_discovery_data`

## Returns
Plot showing different strategies %(hits discovered) vs %(compounds tested)

![alt text](https://github.com/virenar/active_screening/blob/master/performance_active_learner_comparision.png)
