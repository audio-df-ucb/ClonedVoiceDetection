from random import random, sample, seed
import pandas as pd
import numpy as np

# helper function
def loadExistingFile(file_path):
    return pd.read_csv(file_path)

class LJDataLoader:
    # initialization
    def __init__(
        self, data_path: str, id_col: str = "id", filter_cols: list = []
    ) -> None:
        assert ".csv" in data_path, "Data Path should be a csv file."
        self.metadata = pd.read_csv(data_path)
        self._validateData()
        # self._filterCols(filter_cols)
        self.id_col = id_col

    # data validation
    def _validateData(self):
        self.metadata = self.metadata.dropna().reset_index()

    # filtering columns
    def _filterCols(self, filter_cols):
        for col in filter_cols:
            self.metadata = self.metadata[self.metadata[col] == 0]

    # data sampling
    def sample(self, perc: float = 0.1):
        self.metadata = self.metadata.sample(frac=perc, ignore_index=True)

    # splitting data into train, dev, and test sets
    def splitData(
        self, train_perc=0.6, dev_perc=0.2, test_perc=0.2, shuffle: bool = True
    ):
        assert train_perc + dev_perc + test_perc == 1, ""

        if shuffle:
            self.metadata = self.metadata.sample(
                frac=1, ignore_index=True, random_state=12
            )

        self.metadata["type"] = None

        train_idx, dev_idx = int(self.metadata.shape[0] * train_perc), int(
            self.metadata.shape[0] * (train_perc + dev_perc)
        )

        self.metadata.loc[:train_idx, "type"] = "train"
        self.metadata.loc[train_idx:dev_idx, "type"] = "dev"
        self.metadata.loc[dev_idx:, "type"] = "test"

    # selecting random architecture from a list of columns containing architecture names for mixing data
    def selectRandomArchitecture(self, target_col: str, source_cols: list):
        def randomlySelectCols(rw):
            # setting random seed for reproducibility
            # np.random.seed(12)
            rand_idx = np.random.randint(0, len(source_cols))
            return rw[source_cols[rand_idx]]

        self.metadata[target_col] = self.metadata.apply(
            lambda row: randomlySelectCols(row), axis=1
        )

    # generating final dataframe for experiments
    def generateFinalDataFrame(
        self,
        real_col: str,
        fake_cols: list,
        single_id_entry: bool = False,
        balanced: bool = False,
    ):
        agg_cols = [real_col] + fake_cols

        if single_id_entry:
            filter_df = self.metadata[agg_cols].copy()
            multiclass_labels = np.random.randint(
                0, len(agg_cols), filter_df.shape[0]
            ).reshape(filter_df.shape[0], -1)
            chosen_data = np.take_along_axis(
                filter_df.to_numpy(), multiclass_labels, axis=1
            ).squeeze()
            multiclass_labels = multiclass_labels.squeeze()
            labels = np.where(
                multiclass_labels == 0, 0, 1
            )  # in the future, may need to double check that this works for varying column orders
            architectures = [agg_cols[id] for i in multiclass_labels]
            return pd.DataFrame(
                {
                    "path": chosen_data,
                    "label": labels,
                    "multiclass_label": multiclass_labels,
                    "type": self.metadata["type"],
                    "id": self.metadata["id"],
                    "architecture": architectures,
                }
            )

        filter_df = self.metadata[agg_cols + ["type", "id"]].copy()
        output = pd.melt(
            filter_df,
            id_vars=["type", "id"],
            value_vars=agg_cols,
            value_name="path",
            var_name="architecture",
        )
        output["label"] = np.where(output["architecture"] == real_col, 0, 1)
        multiclass_map = {k: v for v, k in enumerate(agg_cols)}
        output["multiclass_label"] = output["architecture"].map(multiclass_map)
        # output = output.drop(columns=['architecture'])
        
        ### balancing code ##
        if balanced:
            seed(4)

            binary_class_labels = output["label"]
            real_indices = list(np.where(binary_class_labels == 0)[0])
            fake_indices = list(np.where(binary_class_labels == 1)[0])

            # Apply random sampling to rebalance data
            # NOTE: currently using equal p(sample) from each all fake samples.
            # E.g. we just random sample from all with a 1 class.
            if len(real_indices) < len(fake_indices):
                fake_indices = sample(fake_indices, len(real_indices))
            elif len(real_indices) > len(fake_indices):
                real_indices = sample(real_indices, len(fake_indices))

            output = output.iloc[real_indices + fake_indices, :].sort_index()

        ### END ###
        return output
