import pandas as pd
import opensmile
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

# base_path
base_path = "/home/ubuntu/"

############################################################################################
# Base Class ###############################################################################
############################################################################################

class smileFeatureSelectorBase:
    # initialize the class to select features
    def __init__(
        self, df, metadata, standardize: bool = True, scaler=StandardScaler()
    ) -> None:
        print("Initializing data...")

        self.data = df
        self.metadata = metadata
        self.all_features = self.data.drop(columns=self.metadata).columns

        self.train_df = self.data[self.data["type"] == "train"].copy()
        self.dev_df = self.data[self.data["type"] == "dev"].copy()
        self.test_df = self.data[self.data["type"] == "test"].copy()

        ## standardize the features inside the train, dev, and test sets for the selected features
        if standardize:
            print("Standardizing features...")
            cols_to_scale = list(self.all_features)
            scaler.fit(self.train_df[cols_to_scale])
            self.train_df.loc[:, cols_to_scale] = scaler.transform(
                self.train_df.loc[:, cols_to_scale]
            )
            self.dev_df.loc[:, cols_to_scale] = scaler.transform(
                self.dev_df.loc[:, cols_to_scale]
            )
            self.test_df.loc[:, cols_to_scale] = scaler.transform(
                self.test_df.loc[:, cols_to_scale]
            )
            self.scaler = scaler
        else:
            self.scaler = None

        # print('smileFeatureSelector object initialized.\n')

############################################################################################
# Feature Selection From Model #############################################################
############################################################################################
class smileFeatureSelectFromModel(smileFeatureSelectorBase):
    def __init__(
        self, df, metadata, standardize: bool = True, model=RandomForestClassifier()
    ):
        """
        Initialize the smileFeatureSelectorBruteForce class.
        """
        # initialize the base class
        super().__init__(df, metadata, standardize)

        # load the model to use for brute force feature selection
        self.model = model
        print("smileFeatureSelectFromModel object initialized.\n")

    # ... (rest of the methods in smileFeatureSelectFromModel)
    def select_features_binary(
        self,
        max_features=10,
        return_df=False,
        print_features=True,
        return_features=False,
    ):
        """
        Selects the top num_features features based on the model specified
        """

        # for binary classification
        sfm_features = self._run_sfm(
            self.train_df, self.dev_df, max_features, multiclass=False
        )
        self.binary_feature_set = set(sfm_features)

        if print_features:
            print("\nSelected features:.\n")
            for count, item in enumerate(self.binary_feature_set):
                print("{}. {}".format(count + 1, item))

        if return_features:
            return list(self.binary_feature_set)

        if return_df:
            return self.data[
                self.data.columns.intersection(
                    self.metadata + list(self.binary_feature_set)
                )
            ], list(self.binary_feature_set)

    def select_features_multiclass(
        self,
        max_features=10,
        archs="all_archs",
        return_df=False,
        print_features=True,
        return_features=False,
    ):
        # for multiclass classification
        sfm_features = self._run_sfm(
            self.train_df, self.dev_df, max_features, multiclass=True
        )
        self.multiclass_feature_set = set(sfm_features)

        if print_features:
            print("\nSelected features:.\n")
            for count, item in enumerate(self.multiclass_feature_set):
                print("{}. {}".format(count + 1, item))

        if return_features:
            return list(self.multiclass_feature_set)

        if return_df:
            return self.data[
                self.data.columns.intersection(
                    self.metadata + list(self.multiclass_feature_set)
                )
            ], list(self.multiclass_feature_set)

    def _run_sfm(self, trdf, dvdf, max_features, multiclass=False):
        # split train data into X and y
        X_train = trdf.drop(columns=self.metadata).copy()
        if multiclass:
            y_train = trdf["multiclass_label"].copy()
        else:
            y_train = trdf["label"].copy()

        # instantiating the model and fitting it
        sfm_model = SelectFromModel(self.model, max_features=max_features)
        sfm_model.fit(X_train, y_train)

        # getting the selected features
        sfm_features = list(X_train.columns[sfm_model.get_support()])
        return sfm_features
        import pandas as pd
