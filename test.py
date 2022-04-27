import time
import os
import pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from featureextraction import extract_features
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import warnings

warnings.filterwarnings("ignore")


# path to training data
source = "it_development_set/"
#source = "en_development_set/"

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


class_y=[]

print("Do you want to Test a Single Audio: Press '1' or The complete Test Audio Sample: Press '0' ?")
take = int(input().strip())


#verifica Test-Set
if take == 0:
    
    #svm Section
    test_file = "it_development_set_test.txt"
    #test_file = "en_development_set_test.txt"
    file_paths = open(test_file, 'r')
    svm_tot_predict=[]
    for path in file_paths:

        path = path.strip()
        print("(SVM)","\n","Testing Audio : ", path)
        sr, audio = read(source + path)
        vector = extract_features(audio, sr)

        svm = svm_models[0]
        predict= svm.predict(vector) #predizione della classe di ogni segmento dell'audio
        
        for j in range(len(predict)):
            class_y.append((path.split("-")[0]))

        for j in range(len(predict)):
            svm_tot_predict.append(predict[j])
        #print("il vincitore è l'utente",user,"con",precision,"% di precisione")    
    #end Svm Section

    #Knn Section
    file_paths = open(test_file, 'r')
    knn_tot_predict=[]
    for path in file_paths:

        path = path.strip()
        print("(KNN)","\n","Testing Audio : ", path)
        sr, audio = read(source + path)
        vector = extract_features(audio, sr)
        knn = knn_models[0]
        predict= knn.predict(vector) #predizione della classe di ogni segmento dell'audio
        for j in range(len(predict)):
            knn_tot_predict.append(predict[j])
    #end Knn Section


    #Gmm Section
    file_paths = open(test_file, 'r')
    gmm_tot_score=[]
    gmm_class_y=[]
    gmm_tot_predict=[]
    for path in file_paths:
        gmm_tot_score=[]
        path = path.strip()
        print("(GMM)","\n","Testing Audio : ", path)
        sr, audio = read(source + path)
        vector = extract_features(audio, sr)
        log_likelihood = np.zeros(len(gmm_models))
        predict_likelihood = np.zeros(len(gmm_models))        
        for i in range(len(gmm_models)):
                gmm = gmm_models[i]  # checking with each model one by one
                scores = np.array(gmm.score(vector))
                log_likelihood[i] = scores.sum()
                frame_score = np.array(gmm.score_samples(vector))
                gmm_tot_score.append(frame_score)                    
        gmm_predict=np.argmax(gmm_tot_score,axis=0)
        for i in range(len(gmm_predict)):
         gmm_tot_predict.append(speakers[gmm_predict[i]])
         gmm_class_y.append((path.split("-")[0]))
    #end Gmm Section

    #Plot Section#
    #model Accuracy
    svm_accuracy=np.round(accuracy_score(class_y,svm_tot_predict)*100,2)
    knn_accuracy=np.round(accuracy_score(class_y,knn_tot_predict)*100,2)
    gmm_accuracy=np.round(accuracy_score(gmm_class_y,gmm_tot_predict)*100,2)
    print("l'accuratezza dei tre modelli è:","\n","(SVM)",svm_accuracy,"%","\n","(KNN)",knn_accuracy,"%","\n","(GMM)",gmm_accuracy,"%")
    #model precision
    svm_precision=np.round(precision_score(class_y,svm_tot_predict,average=None)*100,2)
    knn_precision=np.round(precision_score(class_y,knn_tot_predict,average=None)*100,2)
    gmm_precision=np.round(precision_score(gmm_class_y,gmm_tot_predict,average=None)*100,2)
    print("la precisione dei tre modelli è:","\n","(SVM)",svm_precision,"%","\n","(KNN)",knn_precision,"%","\n","(GMM)",gmm_precision,"%")
    #confusion matrix
    user_list =np.unique(class_y)
    fig, (svm_ax, knn_ax, gmm_ax) = plt.subplots(3,1)
    svm_cm_display = ConfusionMatrixDisplay.from_predictions(class_y,svm_tot_predict, display_labels=user_list, normalize="true", values_format='.2%')
    svm_cm_display.ax_.set_title('SVM Confusion Matrix')
    knn_cm_display = ConfusionMatrixDisplay.from_predictions(class_y,knn_tot_predict, display_labels=user_list, normalize="true", values_format='.2%')
    knn_cm_display.ax_.set_title('K-NN Confusion Matrix')
    gmm_cm_display = ConfusionMatrixDisplay.from_predictions(gmm_class_y,gmm_tot_predict, display_labels=user_list, normalize="true", values_format='.2%')
    gmm_cm_display.ax_.set_title('GMM Confusion Matrix')
    
    svm_ax.plot(user_list, svm_precision, 'o')
    svm_ax.set_ylabel('SVM precision %')
    svm_ax.grid()
    knn_ax.plot(user_list, knn_precision, 'o')
    knn_ax.set_ylabel('KNN precision %')
    knn_ax.grid()
    gmm_ax.plot(user_list, gmm_precision, 'o')
    gmm_ax.set_ylabel('GMM precision %')
    gmm_ax.set_xlabel('utente')
    gmm_ax.grid()
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
    precision=np.round((np.max(counts)/total_frames)*100,2)
    print("il vincitore è l'utente",user,"con",precision,"%")
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
    precision=np.round((np.max(counts)/total_frames)*100,2)
    print("il vincitore è l'utente",user,"con",precision,"%")
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
    