class GiniBinner(object):
    
    report = {}
    na_feat = []
    with_error = []
    
    def __init__(self, max_bins = 6, min_size = 0.05, fast = False, **kwargs):
        self.max_bins = max_bins
        self.min_size = min_size
        self.kwargs = kwargs
        self.fast = fast
        self.report = {}
        self.na_feat = []
        self.with_error = []
    
    def fit_transform(self, X, y):
        
        self.X = X
        self.y = y
        
        assert len(X) == len(y)            
        
        #requirements
        import pandas as pd
        import numpy as np
        import scipy as sp
        from tqdm import tqdm
        import pwlf
        from sklearn.metrics import roc_curve, roc_auc_score
        
        tdf = pd.DataFrame(index=X.index, columns=['target'])
        tdf['target'] = y
        
        df_good = tdf['target'].value_counts()[0]
        df_bad = tdf['target'].value_counts()[1]
        
        for feature in tqdm(X.columns):
            
            try:
            
                df = pd.DataFrame(index=y.index, columns=[feature, 'target'])
                init_len = len(df)
                df[feature] = X[feature]; df['target'] = y
                df_na = df[df[feature].apply(lambda x: np.isnan(x)==True)]
                na_len = len(df_na)
                df.dropna(inplace=True)

                if na_len > (1 - self.min_size)*init_len:
                    self.na_feat.append(feature)
                    continue

                big_na_part = len(df_na) > self.min_size*len(y)
                
                df[feature] = df[feature].apply(lambda x: df[df[feature] != np.inf][feature].max() + 1 if x == np.inf else x)
                df[feature] = df[feature].apply(lambda x: df[df[feature] != -np.inf][feature].min() - 1 if x == -np.inf else x)

                t = df['target']; x = df[feature]
                
                fpr, tpr, thresholds = roc_curve(t, x)

                br = t.value_counts()[1]/len(t)

                #weights
                tpr_lag, fpr_lag = np.zeros((1, len(thresholds)))[0], np.zeros((1, len(thresholds)))[0]
                tpr_lag[1:], fpr_lag[1:] = tpr[:-1], fpr[:-1]
                weights = br*abs(tpr - tpr_lag) + (1 - br)*abs(fpr - fpr_lag)

                #bottom compaction
                i = 0
                bot_edge_vol = 0
                for k in range(len(weights)):
                    bot_edge_vol += weights[k]
                    if bot_edge_vol > self.min_size:
                        break
                    i += 1
                if i > 0:
                    weights = np.insert(weights, i+1, sum(weights[0:i+1]))

                    for i in range(1, i+1):
                        weights = np.delete(weights, 1)
                    for i in range(1, i):
                        tpr, fpr, thresholds = np.delete(tpr, 1), np.delete(fpr, 1), np.delete(thresholds, 1)

                #upper compaction
                j = -1
                top_edge_vol = 0
                for k in reversed(range(len(weights))):
                    top_edge_vol += weights[k]
                    if top_edge_vol > self.min_size:
                        break
                    j -= 1

                if j < 0:
                    weights = np.insert(weights, j, sum(weights[j:]))
                    for i in range(-j):
                        weights = np.delete(weights, -1)
                    for i in range(-j-1):
                        tpr, fpr, thresholds = np.delete(tpr, -2), np.delete(fpr, -2), np.delete(thresholds, -2)

                if len(tpr) > self.max_bins-1:

                    lfit = pwlf.PiecewiseLinFit(fpr[1:-1], tpr[1:-1])
                    if self.fast == False:
                        res = lfit.fit(self.max_bins - 1, [0, 1], [0, 1], **self.kwargs)
                    else:
                        res = lfit.fitfast(self.max_bins - 1, **self.kwargs)
                    opt_idxs = []
                    for i in range(1, self.max_bins - 1):
                        opt_idxs.append(np.argmin(abs(fpr - res[i])))
                    opt_thresholds = sorted(thresholds[opt_idxs])
        
                else:
                    opt_thresholds = sorted(thresholds[1:-1])

                def transform(x):
                    if np.isnan(x):
                        return 0
                    for i in reversed(range(len(opt_thresholds))):
                        if x >= opt_thresholds[i]:
                            return i+2
                    return 1

                tdf['woe_%s' % feature] = X[feature].apply(lambda x: transform(x))

                q = 0
                if not big_na_part and len(df_na) > 0:
                    br_na = tdf.groupby(by='woe_%s' % feature)['target'].mean()[0]
                    q = tdf.groupby(by='woe_%s' % feature)['target'].mean() - br_na
                    q = abs(q[q!=0])
                    q = q[q==q.min()].index[0]
                    tdf['woe_%s' % feature] = tdf['woe_%s' % feature].apply(lambda x: q if x==0 else x)

                woe_iv = {'woe':{}, 'iv': 0}
                counter = tdf.groupby(by='woe_%s' % feature)['target'].value_counts()

                for i in np.unique(tdf['woe_%s' % feature]):
                    try:
                        goods = counter[i][0]
                    except:
                        goods = 0.001
                    try:
                        bads = counter[i][1]
                    except:
                        bads = 0.001
                    woe = np.log((goods/df_good)/(bads/df_bad))
                    woe_iv['woe'][i] = woe

                for j in np.unique(tdf['woe_%s' % feature]):
                    try:
                        goods = counter[j][0]
                    except:
                        goods = 0.001
                    try:
                        bads = counter[j][1]
                    except:
                        bads = 0.001
                    woe_iv['iv'] += woe_iv['woe'][j]*((goods/df_good) - (bads/df_bad))


                a = np.unique(tdf['woe_%s' % feature])
                def woe_transform(x):
                    for i in a:
                        if x == i:
                            return woe_iv['woe'][i]

                tdf['woe_%s' % feature] = tdf['woe_%s' % feature].apply(lambda x: woe_transform(x))

                gini = abs(roc_auc_score(tdf['target'], tdf['woe_%s' % feature])*2 - 1)

                self.report[feature] = {'opt_thresholds': opt_thresholds,
                                        'woe': woe_iv['woe'],
                                        'a': a,
                                        'q': q,
                                        'iv': woe_iv['iv'],
                                        'gini': gini}
            except:
                self.with_error.append(feature)
                continue
            
        
        return tdf.drop(columns=['target'], axis=1)
    
    def transform(self, X):
        
        self.X = X
        
        #requirements
        import pandas as pd
        import numpy as np
        import scipy as sp
        from tqdm import tqdm
        from sklearn.metrics import roc_curve, roc_auc_score
        
        tdf = pd.DataFrame(index=X.index)
        
        def transform(x):
            if np.isnan(x):
                return 0
            for i in reversed(range(len(opt_thresholds))):
                if x >= opt_thresholds[i]:
                    return i+2
            return 1
        
        for feature in tqdm(X.columns):
            
            if feature not in self.with_error and feature not in self.na_feat:
            
                def woe_transform(x):
                    for i in self.report[feature]['a']:
                        if x == i:
                            return woe[i]

                opt_thresholds = self.report[feature]['opt_thresholds']
                woe = self.report[feature]['woe']
                q = self.report[feature]['q']
                tdf['woe_%s' % feature] = X[feature].apply(lambda x: transform(x))
                tdf['woe_%s' % feature] = tdf['woe_%s' % feature].apply(lambda x: q if x==0 else x)
                tdf['woe_%s' % feature] = tdf['woe_%s' % feature].apply(lambda x: woe_transform(x))
            
        return tdf   