# global packages
import numpy as np
import pandas as pd

# local packages
from packages.SavedFeatureLoader import loadFeatures

# directory to save embeddings to
SAVED_EMBEDDINGS_DIR = "/home/ubuntu/data/wavefake_data/Embeddings/16000KHz"

# helper function to generate Titanet embeddings
def generateTitaNetEmbeddings(model, paths, normalize):
    embeddings = np.array(
        [
            model.get_embedding(file_path).cpu().detach().numpy()[0]
            for file_path in paths
        ]
    )

    if normalize:
        raise NotImplementedError("Normalizing embeddings is not implemented yet")

    return embeddings


class AudioEmbeddingsManager:
    # initialize with model and data
    def __init__(self, model, data) -> None:
        self.model = model
        self.data = data

    # generate the dataframe of embeddings for experiments
    def generateFeatureDf(
        self, normalize: bool = False, regenerate_embeddings: bool = False
    ):
        # generate embeddings and save to disk
        if regenerate_embeddings:
            embeddings_df = pd.DataFrame(self.generateEmbeddings(normalize))

            feature_cols = list(embeddings_df.columns)
            feature_df = pd.concat((self.data, embeddings_df), axis=1)

        # load embeddings from disk
        else:
            feature_df = loadFeatures(self.data.copy(), "titanet")
            feature_cols = list(set(feature_df.columns) ^ set(self.data.columns))

        return feature_df, feature_cols

    # generate embeddings for each audio file
    def generateEmbeddings(self, normalize):
        return generateTitaNetEmbeddings(self.model, self.data["path"], normalize)
