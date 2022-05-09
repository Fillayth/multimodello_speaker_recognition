import pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from featureextraction import extract_features
import warnings
warnings.filterwarnings("ignore")


print("Per lavorare sul dataset inglese (EN) : Premi '1' o per lavorare sul dataset in italiano (ITA) : Premi '0'")
take = int(input().strip())


if take == 0:
 source = "it_development_set/"
 destgmm = 'Speakers_models/ita/Gmm/'
 destsvm = "Speakers_models/ita/Svm/"
 destknn = "Speakers_models/ita/Knn/"
 train_file = "it_development_set_enroll.txt"
if take == 1:
 source   = "en_development_set/"
 destgmm = 'Speakers_models/en/Gmm/'
 destsvm = "Speakers_models/en/Svm/"
 destknn = "Speakers_models/en/Knn/"
 train_file = "en_development_set_enroll.txt"


file_paths = open(train_file, 'r')
n_sample = 8  #numero di file audio per utente nel training set
count = 1
features = np.asarray(()) #vector of features
class_x = np.asarray(()) #Dataset for classificator model and vector of target
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
    if count == n_sample:
     
     if class_x.size == 0:
        class_x= features
     else:    
        class_x=np.vstack((class_x,features))

     for j in range(len(features)):
            class_y.append((path.split("-")[0]))    
      
     #GMM Fit
     gmm = GMM(n_components = 6, covariance_type='full',tol=0.0000001, max_iter=1000, verbose=1)
     gmm.fit(features)
     # dumping the trained gaussian model
     picklefile = path.split("-")[0]+".gmm"
     cPickle.dump(gmm, open(destgmm + picklefile, 'wb'))
     print('+ modeling completed for speaker:', picklefile,
              " with data point = ", features.shape)

             
     features=np.asarray(())
     count=0
     
    count = count + 1       


#SVM Fit
svm = SVC(C=30.0, kernel='rbf', gamma='auto', probability=True, tol=0.001, cache_size=200, class_weight='balanced', verbose=True, max_iter=- 1, decision_function_shape='ovo', break_ties=False, random_state=None)
svm.fit(class_x,class_y)       
picklefile = "svmodel.svm"
cPickle.dump(svm, open(destsvm + picklefile, 'wb')) #dumping the trained svm model


#KNN Fit
knn = KNN(n_neighbors=12, weights="distance", algorithm="auto", leaf_size=30, p=2, metric="minkowski", metric_params=None, n_jobs=None)
knn.fit(class_x,class_y)
picklefile = "knnmodel.knn"
cPickle.dump(knn, open(destknn + picklefile, 'wb')) #dumping the trained knn model

print('\a')
    