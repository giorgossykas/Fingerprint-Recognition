# This file contains a Class for the evaluation and
# display of the results. It can display ROC curves
# for the user data and the poly Dataset or compare
# two images minutiae points.

# The "compare_fingerprints" method takes two images as input,
# extracts their minutiae points and compares their location and
# orientation. It returns the number of matched minutiae points.
# Depending on the threshold of choice (based on the desired FPR)
# a decision is made for a match or not. It can also be used independently
# for a single comparison.

# The "evaluate(X, Y, targetFPR, embedding_model)" method takes as input the images X,
# their labels Y the desired Flase Positive Rate, targetFPR, and the embedding model.
# If an embedding model is passed as an argument then the ROC curve will be calculated
# from comparisons from the embedding model. If only X, Y, targetFPR arguments are passed
# then the ROC curve will automatically be calculated from comparisons of minutiae points
# (the "compare_fingerprints" method is automatically chosen).

import numpy as np
import math
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt

class Evaluator:

    def __init__(self):
        pass

    def compare_fingerprints(self, image1, image2):

        '''
        Gets 2 images as input, calculates the minutiae points
        of each and then returns the number of matched minutiae
        '''

        # Extract minutiae
        FeaturesTerminations1, FeaturesBifurcations1 = extract_minutiae_features(image1,
                                                                                 spuriousMinutiaeThresh=10,
                                                                                 invertImage=False,
                                                                                 showResult=False,
                                                                                 saveResult=False)
        FeaturesTerminations2, FeaturesBifurcations2 = extract_minutiae_features(image2,
                                                                                 spuriousMinutiaeThresh=10,
                                                                                 invertImage=False,
                                                                                 showResult=False,
                                                                                 saveResult=False)
        # Try and change thres_s and thres_d for better results
        count = 0
        thres_s = 18 # Spatial distance threshold
        thres_d = 45 # Angular distance threshold
        sd = [0]
        for i in range(len(FeaturesTerminations1) + len(FeaturesBifurcations1)):

            if i < len(FeaturesTerminations1):
                x1 = FeaturesTerminations1[i].locX
                y1 = FeaturesTerminations1[i].locY
                angle1 = np.array(FeaturesTerminations1[i].Orientation)
            else:
                x1 = FeaturesBifurcations1[i - len(FeaturesTerminations1)].locX
                y1 = FeaturesBifurcations1[i - len(FeaturesTerminations1)].locY
                angle1 = np.array(FeaturesBifurcations1[i - len(FeaturesTerminations1)].Orientation)

            for j in range(len(FeaturesTerminations2) + len(FeaturesBifurcations2)):
                flag1, flag2 = False, False

                if j < len(FeaturesTerminations2):
                    x2 = FeaturesTerminations2[j].locX
                    y2 = FeaturesTerminations2[j].locY
                    angle2 = np.array(FeaturesTerminations2[j].Orientation)
                else:
                    x2 = FeaturesBifurcations2[j - len(FeaturesTerminations2)].locX
                    y2 = FeaturesBifurcations2[j - len(FeaturesTerminations2)].locY
                    angle2 = np.array(FeaturesBifurcations2[j - len(FeaturesTerminations2)].Orientation)

                # Check the spatial distance
                sd = np.sqrt(np.power(x1 - x2, 2) + np.power(y1 - y2, 2))
                if sd <= thres_s:
                    flag1 = True

                # Check the angular/degree distance
                if angle1.shape[0] == 3 and angle2.shape[0] == 3:
                    if np.all(abs(angle1 - angle2) <= thres_d):
                        flag2 = True
                elif angle1.shape[0] == 1 and angle2.shape[0] == 1:
                    if abs(angle1 - angle2) < thres_d:
                        flag2 = True
                else:
                    if np.all(abs(angle1 - angle2) < thres_d + 180) and np.all(abs(angle1 - angle2) > thres_d):
                        flag2 = True
                # Final check
                if flag1 and flag2:
                    count += 1

        return count

    ################## ROC Curve evaluation #####################

    def compute_probs(self, X, Y, embedding_model):
        '''
        Input
            network : current NN to compute embeddings
            X : tensor of shape (m,w,h,1) containing pics to evaluate for CNN or
            X : tensor of shape (m,w,h) containing pics to evaluate for minutiae
            Y : tensor of shape (m,) containing true class

        Returns
            probs : array of shape (m,m) containing distances
        '''

        if embedding_model != None:  # embedding_model==None means minutiae evaluation, else CNN
            embeddings = embedding_model.predict(X)

        m = X.shape[0]
        nbevaluation = int(m * (m - 1) / 2)
        probs = np.zeros((nbevaluation))
        y = np.zeros((nbevaluation))

        k = 0

        # For each image in the evaluation set
        for i in range(m):
            # Against all other images
            for j in range(i + 1, m):
                # compute the probability of being the right decision
                # it should be 1 for right class, 0 for all other classes

                # embedding_model!=None means CNN so embeddings
                # embedding_model==None means minutiae so compare_fingerprints()
                if embedding_model == None:
                    probs[k] = self.compare_fingerprints(X[i], X[j])
                else:
                    probs[k] = -np.linalg.norm(embeddings[i, :] - embeddings[j, :])

                if Y[i] == Y[j]:
                    y[k] = 1
                else:
                    y[k] = 0
                k += 1

        return probs, y

    def compute_metrics(self, probs, yprobs):
        '''
        Returns
            fpr : Increasing false positive rates such that element i is the false positive rate
                  of predictions with score >= thresholds[i]
            tpr : Increasing true positive rates such that element i is the true positive rate
                  of predictions with score >= thresholds[i].
            thresholds : Decreasing thresholds on the decision function used to compute fpr and tpr.
                         thresholds[0] represents no instances being predicted and is arbitrarily
                         set to max(y_score) + 1
            auc : Area Under the ROC Curve metric
        '''

        # Calculate AUC
        auc = roc_auc_score(yprobs, probs)
        # calculate roc curve
        fpr, tpr, thresholds = roc_curve(yprobs, probs)

        return fpr, tpr, thresholds, auc

    def draw_roc(self, fpr, tpr, thresholds, auc, targetFPR):

        # find threshold
        targetfpr = targetFPR
        _, idx = self.find_nearest(fpr, targetfpr)
        threshold = thresholds[idx]
        recall = tpr[idx]

        # plot no skill
        plt.plot([0, 1], [0, 1], linestyle='--')
        # plot the roc curve for the model
        plt.plot(fpr, tpr, marker='.')
        plt.title('AUC: {0:.3f}\nSensitivity : {2:.1%} @FPR={1:.0e}\nThreshold={3})'.format(auc, targetfpr, recall,
                                                                                            abs(threshold)))
        # show the plot
        plt.show()

    def find_nearest(self, array, value):

        idx = np.searchsorted(array, value, side="left")

        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
            return array[idx - 1], idx - 1
        else:
            return array[idx], idx

    def evaluate(self, X, Y, targetFPR, embedding_model=None):
        '''
        X: images
        Y: labels
        targetFPR: Desired False Positive Rate to get respective threshold
        embedding_model: if passed then it will be used for the ROC curve, otherwise
        if left blank the minutiae points will be used by default.
        
        Returns: Displays the ROC curve of images X based on minutiae or embeddings.
        '''
        probs, yprob = self.compute_probs(X, Y, embedding_model=embedding_model)
        fpr, tpr, thresholds, auc = self.compute_metrics(probs, yprob)
        self.draw_roc(fpr, tpr, thresholds, auc, targetFPR)