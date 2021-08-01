# the-vibes-algorithm

Please cite as
Fatih Aydın, Zafer Aslan, The Construction of a Majority-Voting Ensemble Based on the Interrelation and Amount of Information of Features, The Computer Journal, Volume 63, Issue 11, November 2020, Pages 1756–1774, https://doi.org/10.1093/comjnl/bxz118



We introduced a new ensemble learning algorithm called Vibes, which is better in terms of performance when compared to 85 machine learning algorithms in the WEKA tool. This new algorithm is based on three major processes: (i) assuming whether features are dependent on or independent of each other, (ii) computing the amount of information of features when it is assumed that they are dependent on each other and then sorting them in a descending manner based on the amount of information, (iii) speeding up the algorithm by optimizing the forward search algorithm that is used in the construction of the final hypothesis from base learner hypotheses. As a result of these processes, it has been seen in the experiments that choosing the relevant assumption can boost learning performance if features are independent of each other; considering features according to the amount of information provides high accuracy and diversity of base learner models. According to experiment results, the algorithm that has been developed has the highest average classification accuracy rate across the 33 datasets. The highest and the lowest average classification accuracy rates are 89.80 and 78.03%, respectively.

The datasets and algorithms used in the experiments and experiment results have been shared at the link (https://yadi.sk/d/g0A2RRhoGTrA1g) as .arff (WEKA) and .mat (MATLAB) file formats.

