import sys
import random
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

def get_features_and_responses(data):
    #return x, y = features, response
    x = data[data.columns[1:]]
    y = []
    for i in data['I']:
        if i == "I":
            y.append((1,0))
        else:
            y.append((0,1))
    return x, y

def get_model_performance(data, batch_size, strategy, seed):
    #return a tuple [%chemical_space, %hit_discovery] for each batch iterations
    print "Loading Features:"
    x, y = get_features_and_responses(data)
    print "Spliting the data:"
    x_train, x_test, y_train, y_test, indices_train, indices_test = train_test_split(x,y,range(data.shape[0]), test_size=1-(float(batch_size)/100), random_state = seed)

    print "Starting iterations:"
    picks = (batch_size * data.shape[0])/100 #number of picks of small molecules at each iteration
    hits = len([item for item in y_train if item[1] == 1]) #hits stores number of hits for each iteraction
    total_hits = len([item for item in y if item[1] == 1]) #total small molecules hits for a given target
    result = [] #tuple that stores %discovered hits and %chemical space tested during each iteration
    itr = 1
    for i in range(0,100/batch_size):
        indices_train, indices_test, hits = active_learning(x_train, y_train, x_test, y_test, indices_train, indices_test, strategy, picks,seed, total_hits, hits)

        #measure performance
        percent_chemical_space = i*batch_size
        percent_hits = (hits*100)/total_hits
        result.append([percent_chemical_space,percent_hits])
        print "Iterations: ", itr, strategy.__name__, "|Percent chemical space: ", percent_chemical_space, "|Percent hits: ", percent_hits, "|hits: ", hits

        #new x_train, y_train, and x_test
        x_train = x.ix[indices_train]
        y_train = [y[item] for item in indices_train]
        x_test = x.ix[indices_test]
        y_test = [y[item] for item in indices_test]
        picks = min(len(y_test), picks)
        itr += 1
    print "------"
    return result

def active_learning(x_train,y_train,x_test, y_test, indices_train, indices_test, strategy, picks,seed, total_hits, hits):
    #return the indices of new train and test data according to strategy
    if strategy.__name__ == "get_optimal_sorted_indices":
        indices_picked = get_optimal_sorted_indices(y_test, indices_test)[:picks]
    else:
        df = RandomForestClassifier(n_estimators=12, criterion='gini', bootstrap=True, oob_score=False, n_jobs=4, random_state = seed, verbose=0, class_weight='auto')
        df.fit(x_train,y_train)
        if strategy.__name__ == "get_max_entropy_sorted_indices":
            tup = df.predict_proba(x_test)[1]
        elif strategy.__name__ == "get_max_prob_sorted_indices":
            tup = df.predict_proba(x_test)[0]
        else:
            tup = df.predict(x_test)
        indices_picked = strategy(tup, indices_test)[:picks]

    hits = len([item for item in y_train if item[1] == 1])
    indices_test = [item for item in indices_test if item not in indices_picked]
    indices_train.extend(indices_picked)
    return indices_train, indices_test, hits

def get_random_indices(tuples, indices):
    #return indices of test_set with random shuffle
    random.seed(0)
    random.shuffle(indices)
    return indices

def get_max_entropy_sorted_indices(tuples, indices):
    #return indices of test_set according to max entropy - decending
    entropy = []
    for i in tuples:
        entropy.append((i[0] * np.log(i[0])) + (i[1] * np.log(i[1])))
    sorted_index = np.argsort(entropy)
    return np.array(indices)[sorted_index]

def get_max_prob_sorted_indices(tuples, indices):
    #return indices of test_set according to max prob - decending
    prob = []
    for i in tuples:
        prob.append(i[1])
    sorted_index = np.argsort(prob)
    return np.array(indices)[sorted_index]

def get_optimal_sorted_indices(tuples, indices):
    #return indices of test_set according to optimal picks
    optimal = []
    for i in tuples:
        optimal.append(i[1])
    sorted_index = np.argsort(optimal)[::-1]
    return np.array(indices)[sorted_index]

def plot_model_comparision(random_tup1, max_entropy_tup2, max_prob_tup3, optimal_tup4, outfile):
    #output a png file displaying the performance of different strategies
    random_res1 = zip(*random_tup1)
    max_entropy_res2 = zip(*max_entropy_tup2)
    max_prob_res3 = zip(*max_prob_tup3)
    optimal_res4 = zip(*optimal_tup4)
    #plot lines
    plt.plot(random_res1[0],random_res1[1], "k--", label = "Random")
    plt.plot(max_entropy_res2[0],max_entropy_res2[1],'bo-', label = "Max Entropy")
    plt.plot(max_prob_res3[0], max_prob_res3[1],'go-', label = "Max Probability")
    plt.plot(optimal_res4[0],optimal_res4[1],'r--', label = "Optimal")
    #add legend
    legend = plt.legend(loc="lower right")
    frame = legend.get_frame()
    #label axis
    plt.xlabel("Tested Chemical Space (%)")
    plt.ylabel("Discovered Hits (%)")
    #save figure
    plt.savefig(outfile)


print "Loading pickle data object"
#data = pd.read_csv(sys.argv[1])
#data.to_pickle("drug_discovery_data")
data = pd.read_pickle(sys.argv[1])
random_tup1 = get_model_performance(data, 5, get_random_indices, 0)
max_entropy_tup2 = get_model_performance(data, 5, get_max_entropy_sorted_indices, 1)
max_prob_tup3 = get_model_performance(data, 5, get_max_prob_sorted_indices, 1)
optimal_tup4 = get_model_performance(data, 5, get_optimal_sorted_indices, 0)

outfile = "performance_active_learner_comparision.png"
plot_model_comparision(random_tup1, max_entropy_tup2, max_prob_tup3, optimal_tup4, outfile)
