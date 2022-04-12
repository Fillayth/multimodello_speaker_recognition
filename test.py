import time
import os
import pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from featureextraction import extract_features
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import warnings
warnings.filterwarnings("ignore")


# path to training data
source = "it_development_set/"

# path where training speakers will be saved
destgmm = 'Speakers_models/Gmm/'
destsvm = "Speakers_models/Svm/"
destknn = "Speakers_models/Knn/"


#get files
gmm_files = [os.path.join(destgmm, fname) for fname in
             os.listdir(destgmm) if fname.endswith('.gmm')]
svm_files = [os.path.join(destsvm, fname) for fname in
             os.listdir(destsvm) if fname.endswith('.svm')]
knn_files = [os.path.join(destknn, fname) for fname in
             os.listdir(destknn) if fname.endswith('.knn')]


# Load the Models
gmm_models = [cPickle.load(open(fname, 'rb')) for fname in gmm_files]
svm_models = [cPickle.load(open(fname, 'rb')) for fname in svm_files]
knn_models = [cPickle.load(open(fname, 'rb')) for fname in knn_files]



speakers = [fname.split("/")[-1].split(".gmm")[0] for fname
            in gmm_files]

error = 0
total_sample = 0.0
class_y=[]

print("Do you want to Test a Single Audio: Press '1' or The complete Test Audio Sample: Press '0' ?")
take = int(input().strip())


#verifica Test-Set
if take == 0:
    #svm Section
    test_file = "it_development_set_test.txt"
    file_paths = open(test_file, 'r')
    svm_accuracy_vec=[]
    usernames=[]
    svm_tot_predict=[]
    
    for path in file_paths:

        path = path.strip()
        print("(SVM)","\n","Testing Audio : ", path)
        sr, audio = read(source + path)
        vector = extract_features(audio, sr)

        svm = svm_models[0]
        predict= svm.predict(vector) #predizione della classe di ogni segmento dell'audio
        total_frames=predict.size #numero totale di segmenti

        for j in range(len(predict)):
            class_y.append((path.split("-")[0]))

        uniques, counts  =np.unique(predict, return_counts=True) #numero di occorrenze delle predizioni
        res=dict(zip(counts,uniques))
        user=res.get(np.max(counts))
        accuracy=np.round((np.max(counts)/total_frames)*100,2)
        print("il vincitore è l'utente",user,"con",accuracy,"%")
        
     
        svm_accuracy_vec.append(accuracy)
        usernames.append(user)
        for j in range(len(predict)):
            svm_tot_predict.append(predict[j])
        #end Svm Section

    #Knn Section
    file_paths = open(test_file, 'r')
    knn_accuracy_vec=[]
    usernames=[]
    knn_tot_predict=[]
    for path in file_paths:

        path = path.strip()
        print("(KNN)","\n","Testing Audio : ", path)
        sr, audio = read(source + path)
        vector = extract_features(audio, sr)
        
    
        knn = knn_models[0]
        predict= knn.predict(vector) #predizione della classe di ogni segmento dell'audio
        total_frames=predict.size #numero totale di segmenti
        uniques, counts  =np.unique(predict, return_counts=True) #numero di occorrenze delle predizioni
        res=dict(zip(counts,uniques))
        user=res.get(np.max(counts))
        accuracy=np.round((np.max(counts)/total_frames)*100,2)
        print("il vincitore è l'utente",user,"con",accuracy,"%")

        knn_accuracy_vec.append(accuracy)
        usernames.append(user)
        for j in range(len(predict)):
            knn_tot_predict.append(predict[j])
    #end Knn Section


    #Gmm Section
    file_paths = open(test_file, 'r')
    for path in file_paths:

        total_sample += 1.0
        path = path.strip()
        print("Testing Audio : ", path)
        sr, audio = read(source + path)
        vector = extract_features(audio, sr)

        log_likelihood = np.zeros(len(gmm_models))

        for i in range(len(gmm_models)):
                gmm = gmm_models[i]  # checking with each model one by one
                scores = np.array(gmm.score(vector))
                log_likelihood[i] = scores.sum()

        winner = np.argmax(log_likelihood)
        print("\tdetected as - ", speakers[winner])

        checker_name = path.split("-")[0]
        if speakers[winner] != checker_name:
                error += 1
        time.sleep(1.0)

    print(error, total_sample)
    accuracy = ((total_sample - error) / total_sample) * 100

    print("The Accuracy Percentage for the current testing Performance with MFCC + GMM is : ", accuracy, "%")
    #end Gmm Section

    #Plot Section
    user_list =np.unique(class_y)
       
    fig, (svm_ax, knn_ax) = plt.subplots(1,2)
    svm_cm_display = ConfusionMatrixDisplay.from_predictions(class_y,svm_tot_predict, display_labels=user_list)
    svm_cm_display.ax_.set_title('SVM Confusion Matrix')
    knn_cm_display = ConfusionMatrixDisplay.from_predictions(class_y,knn_tot_predict, display_labels=user_list)
    knn_cm_display.ax_.set_title('K-NN Confusion Matrix')
    svm_ax.plot(usernames, svm_accuracy_vec, 'o')
    svm_ax.set_ylabel(' Accuracy %')
    svm_ax.set_title(' SVM identification')       
    svm_ax.grid()
    knn_ax.plot(usernames, knn_accuracy_vec, 'o')
    knn_ax.set_ylabel('Accuracy %')
    knn_ax.set_xlabel('utente')
    knn_ax.set_title('KNN identification')
    knn_ax.grid()
    plt.plot()
    plt.show(block=True)


#verifica file singolo
if take == 1:
    #Svm
    print("Enter the File name from Test Audio Sample Collection :")
    path = input().strip()
    print("(SVM)","\n","Testing Audio : ", path)
    
    sr, audio = read(source + path)
    vector = extract_features(audio, sr)
     
    svm = svm_models[0]
    predict= svm.predict(vector) #predizione della classe di ogni segmento dell'audio
    total_frames=predict.size #numero totale di segmenti
    uniques, counts  =np.unique(predict, return_counts=True) #numero di occorrenze delle predizioni
    res=dict(zip(counts,uniques))
    user=res.get(np.max(counts))
    accuracy=np.round((np.max(counts)/total_frames)*100,2)
    print("il vincitore è l'utente",user,"con",accuracy,"%")
    #

    #knn
    print("(KNN)","\n","Testing Audio : ", path)
    sr, audio = read(source + path)
    vector = extract_features(audio, sr)
    
    knn = knn_models[0]
    predict= knn.predict(vector) #predizione della classe di ogni segmento dell'audio
    total_frames=predict.size #numero totale di segmenti
    uniques, counts  =np.unique(predict, return_counts=True) #numero di occorrenze delle predizioni
    res=dict(zip(counts,uniques))
    user=res.get(np.max(counts))
    accuracy=np.round((np.max(counts)/total_frames)*100,2)
    print("il vincitore è l'utente",user,"con",accuracy,"%")
    #

    #gmm

    print("Testing Audio : ", path)
    sr, audio = read(source + path)
    vector = extract_features(audio, sr)

    log_likelihood = np.zeros(len(gmm_models))

    for i in range(len(gmm_models)):
        gmm = gmm_models[i]  # checking with each model one by one
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()

    winner = np.argmax(log_likelihood)
    print("\tdetected as - ", speakers[winner])

    time.sleep(1.0)
    