# global packages
import sys
import pandas as pd
import os
import librosa
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import json

# local packages
from packages.SavedFeatureLoader import loadFeatures
from packages.CadenceUtils import *
from packages.BayesSearch import BayesSearch


class CadenceModelManager:
    def __init__(
        self, data, low_pass_filter_cutoff: int = 10, trunc_window_size: int = 100
    ) -> None:  
        self.data = data
        self.low_pass_filter_cutoff = low_pass_filter_cutoff
        
        # assume fixed sampling rate for all files
        self.sr = sr = librosa.load(self.data["path"][0])[1]

    # generate cadence features
    def generate_features(self, window_size, silence_threshold, paths):

        window_size = int(window_size)

        # Normalise amplitudes
        print("Normalizing amplitudes")
        norm_audio = normalize_audio_amplitudes(paths)

        # Truncate silences
        print("Truncating silences")
        _, _, trunc_audio = truncate_silences(
            norm_audio, window_size, silence_threshold
        )
        # Extract pauses
        print("Extracting pauses")
        pauses = self.run_all_files(
            get_silence, window_size, silence_threshold, trunc_audio
        )

        # Extract pause spreads
        print("Extracting pause spreads")
        silence_spreads = self.run_all_files(
            get_silence_spread, window_size, silence_threshold, trunc_audio
        )

        # Extract amplitude and derivative
        print("Extracting amplitude features")
        amps = self.run_all_files(
            get_amplitude, window_size, silence_threshold, trunc_audio
        )

        # Create dataframe
        print("Creating dataframe")
        features = pd.DataFrame(
            {
                "pause_ratio": [item["ratio_pause_voiced"] for item in pauses],
                "pause_mean": [item["mean_of_silences"] for item in silence_spreads],
                "pause_std": [item["spread_of_silences"] for item in silence_spreads],
                "n_pauses": [item["n_pauses"] for item in silence_spreads],
                "amp_deriv": [item["abs_deriv_amplitude"] for item in amps],
                "amp_mean": [item["mean_amplitude"] for item in amps],
            }
        )

        print("Complete")

        return features

    # run the cadence feature extraction pipeline
    def run_cadence_feature_extraction_pipeline(
        self,
        window_size=None,
        silence_threshold=None,
        data=None,
        scaler=None,
        fill_na=None,
        regenerate_features: bool = False,
    ):
        # feature regeneration block
        if regenerate_features:
            if data is None:
                features = self.generate_features(
                    window_size, silence_threshold, self.data["path"]
                )
                full_df = pd.concat((self.data, features), axis=1)
            else:
                features = self.generate_features(
                    window_size, silence_threshold, data["path"]
                )
                full_df = pd.concat((data, features), axis=1)
            feature_cols = list(features.columns)

        # if features are not being regenerated, load them from the saved features
        # this block is tested and used in pipeline
        else:
            full_df = loadFeatures(self.data.copy(), "cadence")
            feature_cols = list(set(full_df.columns) ^ set(self.data.columns))

        full_df, scaler = self.normalize_data(full_df, feature_cols, scaler=scaler)

        if fill_na is not None:
            full_df = full_df.fillna(fill_na)

        return full_df, feature_cols, scaler

    # data normalization using minmaxscaler
    def normalize_data(self, full_df, feature_cols, scaler=None):
        if scaler is None:
            scaler = MinMaxScaler()
            full_df.loc[
                full_df["type"] == "train", feature_cols
            ] = scaler.fit_transform(
                full_df.loc[full_df["type"] == "train", feature_cols]
            )
            full_df.loc[~(full_df["type"] == "train"), feature_cols] = scaler.transform(
                full_df.loc[~(full_df["type"] == "train"), feature_cols]
            )
        else:
            full_df.loc[:, list(features.columns)] = scaler.transform(
                full_df.loc[:, list(features.columns)]
            )

        return full_df, scaler

    # helper function to run a function on all files
    def run_all_files(self, function, window_size, silence_threshold, truncated_audio):
        results = []
        for item in truncated_audio:
            results.append(
                function(
                    item,
                    window_size,
                    silence_threshold,
                    self.sr,
                    self.low_pass_filter_cutoff,
                )
            )
        return results

    # target function for bayesian optimization
    def target_function(
        self,
        data,
        window_size,
        silence_threshold,
        label_col="label",
        model=DecisionTreeClassifier(random_state=12),
    ):
        features, feature_cols, _ = self.run_cadence_feature_extraction_pipeline(
            window_size,
            silence_threshold,
            data=data,
            fill_na=-1,
            regenerate_features=True,
        )
        X = features[feature_cols]
        y = features[label_col]
        return cross_val_score(model, X, y, cv=10).mean()

    # run target function on a set of parameters
    def run_target_function(self, z, data):
        scores = []
        for i in range(z.shape[0]):
            window_size, silence_threshold = int(z[i, 0]), z[i, 1]
            print(f"Running Params: {window_size}, {silence_threshold}")
            scores.append(self.target_function(data, window_size, silence_threshold))
        return np.array(scores)

    # sample parameters for bayesian optimization
    def sample_params(self, count):
        window_size_mean = 300
        window_size_std = 100
        window_min = 25
        silence_threshold_mean = 0.05
        silence_threshold_std = 0.04
        silence_min = 0.005
        silence_max = 0.2

        window_size = np.random.normal(window_size_mean, window_size_std, count)
        window_size[window_size < window_min] = window_min
        window_size = window_size.astype(int)
        silence_threshold = np.random.normal(
            silence_threshold_mean, silence_threshold_std, count
        )
        silence_threshold[silence_threshold < silence_min] = silence_min
        silence_threshold[silence_threshold > silence_max] = silence_max

        return np.concatenate(
            (window_size.reshape(-1, 1), silence_threshold.reshape(-1, 1)), axis=1
        )

    # run bayesian optimization
    def hyperparam_search(self, n_iter, sample_size, init_ex_count, gp_ex_count):
        search_data = (
            self.data[self.data["type"].isin(["train", "dev"])]
            .sample(sample_size)
            .copy()
            .reset_index()
        )
        search_data.to_csv("/home/ubuntu/search_data.csv", index=False)

        bayes_search = BayesSearch(
            search_data,
            self.run_target_function,
            self.sample_params,
            n_iter=n_iter,
            init_ex_count=init_ex_count,
            gp_ex_count=gp_ex_count,
        )
        params, acc = bayes_search.optimize()
        return params, acc

    # run bayesian optimization and save down the best params
    def hyperparam_search_and_features(
        self,
        output_dir,
        output_name,
        n_iter=25,
        sample_size=300,
        init_ex_count=20,
        gp_ex_count=1000,
    ):
        params, _ = self.hyperparam_search(
            n_iter=n_iter,
            sample_size=sample_size,
            init_ex_count=init_ex_count,
            gp_ex_count=gp_ex_count,
        )
        window_size, silence_threshold = params[0], params[1]

        # save down the best params in a json file
        if os.path.exists(os.path.join(output_dir, "params.json")):
            with open(os.path.join(output_dir, "params.json")) as file:
                params = json.load(file)
        else:
            params = {}

        if "." in output_name:
            output_name = os.splitext(output_name)[0]

        params[output_name] = {
            "window_size": window_size,
            "silence_threshold": silence_threshold,
        }

        with open(os.path.join(output_dir, "params.json"), "w") as file:
            json.dump(params, file)

        fake_data = self.data[self.data["label"] == 1].copy()
        features = self.generate_features(
            window_size, silence_threshold, fake_data["path"]
        )
        full_df = pd.concat((self.data, features), axis=1)
        full_df.to_csv(os.path.join(output_dir, f"{output_name}.csv"), index=False)


def save_features(metadata_path, params_json_path):
    pass
