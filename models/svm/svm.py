import random
import feature_extractor as FE
from sklearn import svm
from sklearn.metrics import accuracy_score

class svm:
    def __init__(self):
        ''' initialize svm model ''' 
        self.MODEL = svm.SVC()

        ''' number of training subsets '''
        self.M = 8

        ''' validation variables '''
        self.VA_ACC= [0]*self.M
        self.i_HIGHEST_ACC = 0

        ''' test accuracy '''
        self.TEST_ACC = 0

        ''' initialize genres for each set '''
        self.TR_T = []
        self.VA_T = []
        self.TE_T = []
        
        ''' use feature_extractor to grab data '''
        self.DATA = FE.feature_extractor()

        ''' initialize track ids for each set '''
        self.TR_IDS= self.DATA.get_training_dataset_song_ids()
        self.VA_IDS = self.DATA.get_validation_dataset_song_ids()
        self.TE_IDS = self.DATA.get_test_dataset_song_ids()

        ''' initialize each feature set '''
        self.TR_X = []
        self.VA_X = []
        self.TE_X = []

        ''' initialize flags '''
        self.FEATURES_READY = False
        self.TEST_READY = False

    def __prepare_features_and_genres(self, stat=KURTOSIS):
        ''' Fills sets X and T for training, validation and test data '''

        if self.FEATURES_READY:
            print('Features and genres ready for cross-validation. Cancelling method.')
            return
        
        ''' Getting training track features and genres '''
        TR_FULL = self.get_features_and_genres(self.TR_IDS,stat)
        VA_FULL = self.get_features_and_genres(self.VA_IDS,stat)
        TE_FULL = self.get_features_and_genres(self.TE_IDS,stat)

        ''' shuffle order of tracks in each set '''
        random.shuffle(TR_FULL)
        random.shuffle(VA_FULL)
        random.shuffle(TE_FULL)

        ''' separate features from genre for each set '''
        self.fill_X_and_T(TR_FULL, self.TR_X, self.TR_T)
        self.fill_X_and_T(VA_FULL, self.VA_X, self.VA_T)
        self.fill_X_and_T(TE_FULL, self.TE_X, self.TE_T)

        return

    def __get_features_and_genres(self, ids, stat):
        ''' Gets features data and genre for set based on stat '''
        X = []
        for i in ids:
            x = []
            for j in feature_extractor.feature_type:
                x.append(i,j,stat)
            x.append(self.DATA.get_genre(i))
            X.append(x)
        return X

    def __fill_X_and_T(self, full, x, t):
        ''' Sets X and T for each set '''
        for i in range(len(full)):
            x.append(full[i][0:-1])
            t.append(full[i][-1])
        return

    def __cross_validate(self):
        ''' Performs M-fold cross-validation for best SVM model '''

        if not self.FEATURES_READY:
            print('Features are not ready. Cancelling method.')
            return
        
        ''' reset these each time we run method'''
        self.VA_ACC = [0]*self.M
        self.i_HIGHEST_ACC = 0

        for i in range(self.M):
            TR_subset_X, TR_subset_T = self.make_TR_subset(i)
            curr_model = svm.SVC()
            curr_model.fit(TR_subset_X, TR_subset_T)
            CL = curr_model.predict(self.VA_X)
            self.VA_ACC[i] = accuracy_score(self.VA_Y,CL)
            if self.VA_ACC[i] >= self.VA_ACC[self.i_HIGHEST_ACC]:
                self.MODEL = curr_model
                self.i_HIGHEST_ACC = i
        return

    def __make_TR_subset(self, i):
        ''' returns subset of training set based on cross-validation iteration'''
        N = len(self.TR_X)
        subset_N = N/self.M
        X = self.TR_X[i*subset_N:(i+1)*subset_N]
        t = self.TR_T[i*subset_N:(i+1)*subset_N]
        return X, t

    def __test_model(self):

        if not self.TEST_READY:
            print('Cross-validation not performed yet. Cancelling method.')
            return

        CL = self.MODEL.predict(self.TE_X)
        self.TEST_ACC = self.accuracy_eval(CL, self.TE_T)
        return

    def __get_VA_ACC_RESULTS(self):
        return self.VA_ACC
    
    def __get_index_highest_VA_ACC(self):
        return self.i_LOWEST_ERR
    
    def __get_TEST_ACC(self):
        return self.TEST_ACC
    
