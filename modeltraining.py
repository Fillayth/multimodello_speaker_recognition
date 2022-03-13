import pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture as GMM
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from featureextraction import extract_features
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# path to training data
# source   = "development_set/"
source = "development_set/"

# path where training speakers will be saved
destgmm = 'Speakers_models/Gmm/'
destsvm = "Speakers_models/Svm/"
destknn = "Speakers_models/Knn/"
train_file = "development_set_enroll.txt"
file_paths = open(train_file, 'r')

i=0
j=0
count = 1


#vector of features
features = np.asarray(())
#Dataset for classificator model and vector of target
class_x =np.asarray(())
class_y=[]


# Extracting features for each speaker (5 files per speakers)
for path in file_paths:
    path = path.strip()
    print(path)

    # read the audio
    sr, audio = read(source + path)

    # extract 40 dimensional MFCC & delta MFCC features
    vector = extract_features(audio, sr)

    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))
    
    
     #a differenza del vettore utile alla produzione dei gmm (ricreato per ogni persona) per i classificatori si genera un un'unico dataset
     #assegnando ad ogni feature proveniente dallo stesso soggetto una classe numerica crescente
    if count == 5:
     
     if class_x.size == 0:
        class_x= features
     else:    
        class_x=np.vstack((class_x,features))

     for j in range(len(features)):
            class_y.append((i))    
      
     #GMM Fit
     gmm = GMM(n_components = 16, covariance_type='diag',n_init = 3)
     gmm.fit(features)
     # dumping the trained gaussian model
     picklefile = path.split("-")[0]+".gmm"
     cPickle.dump(gmm, open(destgmm + picklefile, 'wb'))
     print('+ modeling completed for speaker:', picklefile,
              " with data point = ", features.shape)

     i=i+1         
     features=np.asarray(())
     count=0
     
    count = count + 1       

        
#SVM Fit
svm = SVC(C=100.0, kernel='rbf', gamma='auto', shrinking=True, probability=True, tol=0.001, cache_size=200, 
class_weight=None, verbose=True, max_iter=- 1, decision_function_shape='ovo', break_ties=False, random_state=None)
svm.fit(class_x,class_y)       
#dumping the trained svm model
picklefile = "svmodel.svm"
cPickle.dump(svm, open(destsvm + picklefile, 'wb'))




#KNN Fit
knn = KNN()
knn.fit(class_x,class_y)
#dumping the trained knn model
picklefile = "knnmodel.knn"
cPickle.dump(knn, open(destknn + picklefile, 'wb'))


    