import pandas as pd
import numpy as np
import os
import random
from tqdm import tqdm
from pathlib import Path
import opensmile

# base_path
base_path = "/home/ubuntu"


class smileFeatureGenerator:
    # initiate the class to generate openSMILE ComParE_2016 functionals
    def __init__(
        self,
        data_path: str,
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    ) -> None:
        self.data_path = data_path
        self.feature_extractor = opensmile.Smile(
            feature_set=feature_set, feature_level=feature_level
        )
        # store the wav files in a list
        self.wav_list = self._getWavList()
        assert len(self.wav_list) > 0, "No wav files found in data path"

    # private method to iterate through the data path and store the wav files in a list
    def _getWavList(self):
        wav_list = []
        for file_name in os.listdir(self.data_path):
            if ".wav" in file_name.lower():
                wav_list.append(file_name)
        return wav_list

    # generate openSMILE features
    def generateFeatures(self):
        print("Generating openSMILE features...\n")

        self.smile_df = pd.DataFrame()

        for i in tqdm(range(len(self.wav_list))):
            file_path = os.path.join(self.data_path, self.wav_list[i])
            try:
                features = self.feature_extractor.process_file(file_path).reset_index()
            except:
                print("Error processing file: {}".format(file_path))
                continue

            # compute file duration
            duration = features["end"] - features["start"]
            duration = duration.astype("timedelta64[ms]") / 1000
            features.insert(1, "duration(seconds)", duration)

            features.drop(columns=["start", "end"], inplace=True)

            self.smile_df = pd.concat([self.smile_df, features]).reset_index(drop=True)

        print("\nopenSMILE features generated... call saveFeatures(filename)\n")

    # save the feature to disk for loading during experiments
    def saveFeatures(self, filename: str):
        self.smile_df.to_csv(filename, index=False)
        print("Features saved to {}\n".format(filename))
