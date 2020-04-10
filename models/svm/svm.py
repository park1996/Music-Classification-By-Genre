import feature_extractor as FE
from sklearn import svm as SVM_MODEL
from sklearn.metrics import accuracy_score
import time

class svm:
    def __init__(self):

        start_time = time.time()
        
        ''' initialize svm model ''' 
        self.MODEL = SVM_MODEL.SVC()
        ''' checks if model is trained '''
        self.TRAINED = False

        ''' accuracy from validation '''
        self.HIGHEST_VA_ACC = 0

        ''' test accuracy '''
        self.TEST_ACC = 0

        ''' initialize each feature set '''
        self.TR_X = []
        self.VA_X = []
        self.TE_X = []

        ''' initialize each genre set '''
        self.TR_T = []
        self.VA_T = []
        self.TE_T = []

        ''' stores validation accuracy of each trained model '''
        self.ALL_VA_ACC = []
        
        ''' use feature_extractor to grab data '''
        self.DATA = FE.feature_extractor()

        ''' initialize track ids for each set '''
        self.TR_IDS= self.DATA.get_training_dataset_song_ids()
        self.VA_IDS = self.DATA.get_validation_dataset_song_ids()
        self.TE_IDS = self.DATA.get_test_dataset_song_ids()      

        print ('Elapsed time to initialize: ' + str(time.time() - start_time) + ' seconds\n')
        return

    def prepare_data(self, stat=FE.statistic_type.KURTOSIS):
        ''' Fills sets X and T for training, validation and test data '''
        
        print('Preparing training data...')
        start_time = time.time()
        self.get_features_and_genres(self.TR_IDS, stat, self.TR_X, self.TR_T)
        print ('Elapsed time to get training data: ' + str(time.time() - start_time) + ' seconds\n')
        
        print('Preparing validation data...')
        start_time = time.time()
        self.get_features_and_genres(self.VA_IDS, stat, self.VA_X, self.VA_T)
        print ('Elapsed time to get validation data: ' + str(time.time() - start_time) + ' seconds\n')
        
        print('Preparing test data...')
        start_time = time.time()
        self.get_features_and_genres(self.TE_IDS, stat, self.TE_X, self.TE_T)
        print ('Elapsed time to get test data: ' + str(time.time() - start_time) + ' seconds\n')

        return

    def get_features_and_genres(self, ids, stat, X, T):
        ''' Gets features data and genre for set based on stat '''
        for i in ids:
            x = []
            for j in FE.feature_type:
                features = self.DATA.get_feature(i,j,stat)
                for k in features:
                    x.append(k)
            X.append(x)
            T.append(self.DATA.get_genre(i))
        return

    def train_and_validate(self, k, d=3, g='scale'):
        ''' Trains SVM model and then validates model '''
        
        if len(self.TR_X) == 0 or len(self.TR_T) == 0 or len(self.VA_X) == 0 or len(self.VA_T) == 0:
            print('Call prepare_data(...) first. Cancelling method.')
            return

        print('Training now...')
        start_time = time.time()
        curr_acc = 0
        curr_model = SVM_MODEL.SVC(kernel=k, degree=d, gamma=g)
        curr_model.fit(self.TR_X, self.TR_T)
        print('Using validation data...')
        CL = curr_model.predict(self.VA_X)
        curr_acc = accuracy_score(self.VA_T,CL)
        print('Accuracy of model: ' + str(curr_acc))
        if curr_acc >= self.HIGHEST_VA_ACC:
            print('This model has the best performance so far.')
            self.MODEL = curr_model
            self.HIGHEST_VA_ACC = curr_acc
        self.ALL_VA_ACC.append(curr_acc)

        print ('Elapsed time for training and validation: ' + str(time.time() - start_time) + ' seconds\n')
        self.TRAINED = True
        return

    def test_model(self):

        if not self.TRAINED:
            print('Model not trained yet. Call train_and_validate(). Cancelling method.')
            return
        
        start_time = time.time()
        CL = self.MODEL.predict(self.TE_X)
        self.TEST_ACC = accuracy_score(self.TE_T,CL)
        print ('Elapsed time for testing: ' + str(time.time() - start_time) + ' seconds\n')
        return  

    def get_best_validation_accuracy(self):
        return self.HIGHEST_VA_ACC
    
    def get_test_accuracy(self):
        return self.TEST_ACC
