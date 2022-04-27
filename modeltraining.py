import pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from featureextraction import extract_features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings("ignore")

# path to training data
#source   = "en_development_set/"
source = "it_development_set/"

# path where training speakers will be saved
destgmm = 'Speakers_models/Gmm/'
destsvm = "Speakers_models/Svm/"
destknn = "Speakers_models/Knn/"
#train_file = "en_development_set_enroll.txt"
train_file = "it_development_set_enroll.txt"
file_paths = open(train_file, 'r')

i=0
j=0
count = 1


#vector of features
features = np.asarray(())
#Dataset for classificator model and vector of target
class_x = np.asarray(())
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
            class_y.append((path.split("-")[0]))    
      
     #GMM Fit
     gmm = GMM(n_components = 20, covariance_type='full',tol=0.0000001, max_iter=1000, verbose=1)
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

#one hot encoder
#label_encoder =LabelEncoder()
#integer_encoded = label_encoder.fit_transform(class_y)        
#onehot_encoder = OneHotEncoder(sparse=False)
#integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
#onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
#print(onehot_encoded)


#SVM Fit
svm = SVC(C=10.0, kernel='rbf', gamma='auto', probability=True, tol=0.001, cache_size=200, 
class_weight='balanced', verbose=True, max_iter=- 1, decision_function_shape='ovo', break_ties=False, random_state=None)
svm.fit(class_x,class_y)       
#dumping the trained svm model
picklefile = "svmodel.svm"
cPickle.dump(svm, open(destsvm + picklefile, 'wb'))


#KNN Fit
knn = KNN(n_neighbors=10, weights="uniform", algorithm="auto", leaf_size=30,
p=2, metric="minkowski", metric_params=None, n_jobs=None)
knn.fit(class_x,class_y)
#dumping the trained knn model
picklefile = "knnmodel.knn"
cPickle.dump(knn, open(destknn + picklefile, 'wb'))


    