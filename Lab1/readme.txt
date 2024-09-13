This is the readme file to the files songs_to_classify.csv and
training_data.csv.

The files contain the (unlabeled) test data and the (labeled)
training data, respectively.

The columns represent features, as specified by the header and
documented in the instructions. The column "label"
(training_data.csv. only) is encoded as 1 = like, 0 = dislike.

The files can be loaded into Python using, e.g., panda as

import pandas as pd
training=pd.read_csv('training_data.csv', sep=',')

The files were created by Andreas Svensson in August 2018 using
Spotipy (https://spotipy.readthedocs.io/)