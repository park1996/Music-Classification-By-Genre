import soundfile as sf

class audio_preprocessor:
    def __init__(self):
        ''' Constructor for this class '''

    def audio_read(self, input_path, output_path):
        ''' Read audio wav fie and normalize to [-1, 1)  '''
        ''' input_path - input wav file '''
        ''' output_path - output wav file '''
        nbits = 32
        x, samplerate = sf.read(input_path)
        print (x.dtype)
        if x.dtype == "float32":
            audio = x
        else:
            # change range to [-1,1)
            if x.dtype == "uint8":
                nbits = 8
            elif x.dtype == "int16":
                nbits = 16
            elif x.dtype == "int32":
                nbits = 32
            elif x.dtype == "float64":
                nbits = 64

            audio = x / float(2 ** (nbits - 1))

        # special case of unsigned format
        if x.dtype == "uint8":
            audio = audio - 1.0

        if audio.ndim > 1:
            audio = audio[:, 0]

        sf.write(output_path, x, samplerate); 

        return samplerate, audio

    def block_audio(x, blockSize, hopSize, fs):
        """
        Generate blocks from a signal
        Parameters:
        x: Signal (likely, numpy array)
        blockSize: Integer
        hopSize: Integer
        fs: Integer
        returns:
        xb: [[ndarray], ...] of shape (numBlocks x blockSize)
        t: ndarray of shape (NumOfBlocks,)
        """

        # allocate memory
        numBlocks = math.ceil(x.size / hopSize)
        xb = np.zeros([numBlocks, blockSize])
        # compute time stamps
        t = (np.arange(0, numBlocks) * hopSize) / fs
        x = np.concatenate((x, np.zeros(blockSize)), axis=0)
        for n in range(0, numBlocks):
            i_start = n * hopSize
            i_stop = np.min([x.size - 1, i_start + blockSize - 1])
            xb[n][np.arange(0, blockSize)] = x[np.arange(i_start, i_stop + 1)]
        return xb, t


    def normalize_audio_files(self, input_directory, output_directory):
        ''' Normalize all audio files in the input directory and save to output '''
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

