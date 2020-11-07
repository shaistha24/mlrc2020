# ML Reproduction Challenge 2020 (MLRC2020)
## TeamPriCop: Robust Anomaly and Backdoor Attack Detection via Differential Privacy
Team PriCop: ML Reproduction Challenge 2020 working on Robust Anomaly Detection and Backdoor Attack Detection via Differential Privacy Paper (https://openreview.net/pdf?id=SJx0q1rtvS)

Authors original code - https://www.dropbox.com/sh/rt8qzii7wr07g6n/AAAbwokv2sfBeE9XAL2pXv_Aa?dl=0
### Author Contributions to be Validated
- [ ] Theoretical explanation.
- [ ] The effectiveness of applying differential privacy to an autoencoder network for both outlier detection and novelty detection.
- [ ] Real-world task - Hadoop file system log anomaly detection
- [ ] Backdoor attack detection extending the idea of outlier detection and apply differential privacy to improve performance

### TODOs
- [ ] unit test or validate the claims
- [ ] document all results - errors, negative results, re-run or re-implementation issues faced and possible corrections, re-run results.
- [ ] Re-implement or test on another similar dataset

### Key Author Claims - Theory and our reviews
1. CONNECTION BETWEEN DIFFERENTIAL PRIVACY AND OUTLIER DETECTION 
- theorem -  number of outliers in the training dataset and the amount of noise to apply
- definitions
- Key Note:  
  - the definition of outliers in the paper is quite general—it does not make any assumptions about how the outliers are generated.
  - Do not make assumptions about whether these outliers are in training or test data.
  - Therefore, claims that the analysis can shed light on detecting various types of anomalies, including but not limited to outlier/novelty detection, backdoor detection, and noisy label detection.
- DP noise is effective towards reducing the number of false negatives, and further improving the overall utility.

**Note**: detailed reviews for each experimental cases in their respective folders.
