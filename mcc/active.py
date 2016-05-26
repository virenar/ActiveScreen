import pylab
import sklearn
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.cross_validation import cross_val_score
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from random import shuffle
import gzip
import sys
# ### Ingest data:

# #### Round 1 data
#data = pd.read_csv(gzip.open('thrombin.gz'),',',header=None)
data = pd.read_pickle(sys.argv[1])
X = data[range(1,len(data.columns))]
Y = data['I']

mix_rates = np.array([2./3, 1./6, 1./6])
df = RandomForestClassifier(n_estimators=12, criterion='gini',
                            bootstrap=True, oob_score=False, n_jobs=4,
                            random_state=0, verbose=0, class_weight='auto')
def entropy(p):
    return sum([-p_val*np.log2(p_val+1e-10) for p_val in p])

def max_prob(p):
    return p[0]

def select_preds(all_preds, mix_rates, n_preds):
    n_picks = [round(p*n_preds) for p in mix_rates]
    if n_preds - sum(n_picks) != 0:
        n_picks[np.argmax(mix_rates)] += n_preds - sum(n_picks)
    preds = set()
    for i,n_curr in enumerate(n_picks):
        len_init = len(preds); j = 0;
        while len(preds) < (len_init + n_curr):
            preds.add(all_preds[i][j]); j+=1
    return preds

def test_active_learner(argtuple):
    X,Y,strategy_list,mix_rates,batches = argtuple
    if mix_rates is None:
        mix_rates = np.array([1./len(strategy_list) for n in range(len(strategy_list))])
    elif sum(mix_rates) != 1:
        mix_rates = np.array(mix_rates)/float(sum(mix_rates))
    skf = StratifiedKFold(Y,n_folds=batches,shuffle=True,random_state=0)
    for test_inds, train_inds in skf: break
    train_inds = set(train_inds); test_inds = set(test_inds);
    curr_hits = [sum(Y.loc[train_inds]=='A')]; mcc = [0]; auc = [0]
    iter_cnt = 0; strategy_ind = 0;
    while len(test_inds) > 0:
        train_X = X.loc[train_inds]; train_Y = Y.loc[train_inds];
        test_X  = X.loc[test_inds] ; test_Y  = Y.loc[test_inds];
        print 'Iter:',iter_cnt, '#TrainSet:', len(train_inds), '#TestSet:', len(test_inds), '#Hits:', curr_hits[-1]
        print train_X.isnull().sum().sum()
        df.fit(train_X.astype(int).values, train_Y)
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(Y, df.predict_proba(X)[:,0], pos_label='A')
        auc.append(sklearn.metrics.auc(fpr,tpr))
        mcc.append(sklearn.metrics.matthews_corrcoef(df.predict(test_X),test_Y))
        n_preds = min(len(test_Y), int(round(len(Y)*(1./batches))))
        all_preds = []
        probs = df.predict_proba(test_X)
        for strategy in strategy_list:
            scores = [strategy(p) for p in probs]
            sort_idx = np.argsort(scores)
            data_idx = test_X.index[sort_idx]
            all_preds.append(data_idx[-n_preds:])
        preds = select_preds(all_preds, mix_rates, n_preds)
        train_inds = train_inds.union(set(preds))
        test_inds.difference_update(preds)
        curr_hits.append(sum(Y.loc[train_inds]=='A'))
        iter_cnt+=1
    return curr_hits, mcc, auc

def get_baseline(Y,batches=20):
    num_hits = sum(Y=='A')
    hits_optimal = np.array([1]*num_hits + [0]*(len(Y)-num_hits))
    hits_random = hits_optimal.copy()
    shuffle(hits_random)
    curr_hits_optimal = []; curr_hits_random = [];
    skf = sklearn.cross_validation.KFold(len(Y),shuffle=False,n_folds=batches)
    for tr,ts in skf:
        curr_hits_optimal.append(sum(hits_optimal[ts]))
        curr_hits_random.append(sum(hits_random[ts]))
    curr_hits_optimal = np.cumsum(curr_hits_optimal)
    curr_hits_random = np.cumsum(curr_hits_random)
    return curr_hits_optimal, curr_hits_random, num_hits

curr_hits_optimal, curr_hits_random, num_hits = get_baseline(Y)
curr_hits_entropy, mcc_entropy, auc_entropy = test_active_learner((X,Y,[entropy],[1],20))
curr_hits_maxprob, mcc_maxprob, auc_maxprob = test_active_learner((X,Y,[max_prob],[1],20))
curr_hits_hybrid, mcc_hybrid, auc_hybrid = test_active_learner((X,Y,[entropy,max_prob],[0.5,0.5],20))
pylab.figure(figsize=(11,8))
pylab.plot(np.array(range(len(curr_hits_entropy)))*100./float(len(curr_hits_entropy)-1),
           np.array(curr_hits_entropy)/float(num_hits)*100, 'ko-',label='Max Entropy')
pylab.plot(np.array(range(len(curr_hits_maxprob)))*100./float(len(curr_hits_maxprob)-1),
           np.array(curr_hits_maxprob)/float(num_hits)*100, 'bo-',label='Max Probability')
pylab.plot(np.array(range(len(curr_hits_optimal)))*100./float(len(curr_hits_optimal)-1),
           np.array(curr_hits_optimal)/float(num_hits)*100, 'g--',label='Optimal')
pylab.plot(np.array(range(len(curr_hits_random)))*100./float(len(curr_hits_random)-1),
           np.array(curr_hits_random) /float(num_hits)*100, 'y--',label='Random')
pylab.plot(np.array(range(len(curr_hits_hybrid)))*100./float(len(curr_hits_hybrid)-1),
           np.array(curr_hits_hybrid) /float(num_hits)*100, 'r--',label='Hybrid')
pylab.xlabel('Tested Chemical Space (%)')
pylab.ylabel('Discovered Hits (%)')
pylab.legend(loc=0)
pylab.rc('font', size=20)
pylab.savefig('active_hits.png',dpi=100)
