class featureextractor:
    def __init__(self, audio_input_file):
        ''' Constructor for this class '''
        self.audio_file = audio_input_file

    def create_spectrogram(self, output_file):
        ''' Save spectrogram to output file '''
        pass

    def get_mfcc(self):
        ''' Return Mel-frequency cepstral coefficients (MFCCs) '''
        pass

    def get_fourier_transform(self):
        ''' Return Fourier transform '''
        pass

    def get_tempo(self):
        ''' Return tempo over time '''
        pass

    def get_spectral_contrast(self):
        ''' Get spectral contrast '''
        pass
