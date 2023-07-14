# global packages
import sys
import os
import multiprocessing
import pandas as pd
import mlflow
import time
import argparse

# local packages
sys.path.append("/home/ubuntu/ClonedVoiceDetection/src")
import packages.ExperimentPipeline as ep
from packages.TIMITDataLoader import TIMITDataLoader
from packages.LJDataLoader import LJDataLoader
from packages.AudioEmbeddingsManager import AudioEmbeddingsManager
from packages.ModelManager import ModelManager
from packages.CadenceModelManager import CadenceModelManager
from packages.SmileFeatureManager import SmileFeatureManager

# fixed values
timit_data_path = "/home/ubuntu/data/TIMIT_and_ElevenLabs/TIMIT and ElevenLabs"
fake_voices = [
    "Adam",
    "Antoni",
    "Arnold",
    "Bella",
    "Biden",
    "Domi",
    "Elli",
    "Josh",
    "Obama",
    "Rachel",
    "Sam",
]
# set the models to run
models = ["logreg", "random_forest"]


# helper functions
def chunks(lst, n):
    # sort the list
    lst.sort()
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# function that runs the pipeline asynchonously
def run_pipeline(
    data_df, open_smile_feature_count, run_name_prefix, run_tags, create_df_artifact
) -> None:
    # create and run pipeline object
    exp = ep.ExperimentPipeline(
        fake_cols=["ElevenLabs"], metadata_path=None, data_df=data_df
    )
    exp.generate_features(
        feature_method="all", open_smile_feature_count=open_smile_feature_count
    )
    exp.train_predict_using_models(
        run_name_prefix=run_name_prefix,
        run_tags=run_tags,
        models=models,
        create_df_artifact=create_df_artifact,
        label_type="label",
    )


# main function
def main(
    experiment_name,
    open_smile_feature_count,
    create_df_artifact,
    num_processes,
    save_path,
):
    # start timing
    start_time = time.time()

    print("\nRunning pipeline for experiment: \n", experiment_name)
    mlflow.set_experiment(experiment_name)

    print("\nopen_smile_feature_count: \n", open_smile_feature_count)
    print("\ncreate_df_artifact: \n", create_df_artifact)
    print("\nnum_processes: \n", num_processes)
    print(
        "\nusing {} processes out of {} available processes: \n".format(
            num_processes, multiprocessing.cpu_count()
        )
    )

    # load the timit data
    timit_data_loader = TIMITDataLoader(timit_data_path)
    # generate the split
    df = timit_data_loader.generate_split()
    # get speakers
    df["speaker"] = [item.split("/")[-1].split("_")[0] for item in df["path"]]

    # create partitions
    real_speakers = list(
        set([item for item in df["speaker"] if not item.startswith(tuple(fake_voices))])
    )
    fake_speakers = list(
        set([item for item in df["speaker"] if item.startswith(tuple(fake_voices))])
    )

    real_speaker_partitions = list(chunks(real_speakers, 20))
    fake_speaker_partitions = list(chunks(fake_speakers, 2))

    ####################################
    ##### start mutliprocessing ########
    ####################################

    # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=num_processes)

    # list for holding task arguments
    task_args = []

    ######################################
    ############# create tasks ###########
    ######################################

    # counter for labeling runs
    counter = 1

    # loop through the partitions to remove voices
    for fake_speaker_chunk in fake_speaker_partitions:
        for real_speaker_chunk in real_speaker_partitions:
            # voices to remove
            voices_to_remove = fake_speaker_chunk + real_speaker_chunk

            # re-instantiate the loader
            timit_data_loader = TIMITDataLoader(timit_data_path)

            # generating split speaker test from the
            data_df = timit_data_loader.generate_split_speaker(
                voices_to_remove, folder=False
            )

            # other task arguments
            run_name_prefix = f"multivoice_run_{counter}"
            run_tags = {"voices_to_remove": voices_to_remove}

            # arguments for the task
            args = (
                data_df,
                open_smile_feature_count,
                run_name_prefix,
                run_tags,
                create_df_artifact,
            )

            task_args.append(args)

            counter += 1

    ####################################
    ##### run multiprocessing ##########
    ####################################

    # run the pipeline in parallel
    pool.starmap_async(run_pipeline, task_args)

    # close the pool and wait for the work to finish
    pool.close()
    pool.join()

    ####################################
    ####### aggregate results ##########
    ####################################

    # get all the runs for the experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    runs = mlflow.search_runs(experiment_ids=experiment_id)

    # aggregate results and save to csv
    agg_results = (
        runs.groupby(["tags.feature_method", "tags.estimator_name", "tags.label_type"])[
            "metrics.accuracy",
            "metrics.0_accuracy",
            "metrics.1_accuracy",
            "metrics.eer_score",
        ]
        .mean()
        .reset_index()
    )
    new_column_names = {
        "tags.feature_method": "feature_method",
        "tags.estimator_name": "estimator_name",
        "tags.label_type": "label_type",
        "metrics.accuracy": "accuracy",
        "metrics.0_accuracy": "real_accuracy",
        "metrics.1_accuracy": "fake_accuracy",
        "metrics.eer_score": "eer_score",
    }
    if save_path.lower().endswith(".csv"):
        agg_results.to_csv(save_path)
    else:
        agg_results.to_csv(save_path + f"/results_{experiment_name}.csv", index=False)

    print("\nAggregated results saved to: \n", save_path)

    ####################################
    ######### end the script ###########
    ####################################

    # end timing
    end_time = time.time()
    execution_time_seconds = end_time - start_time

    # convert to minutes
    execution_time_minutes = execution_time_seconds / 60

    print("\nAll async pipeline runs complete \n")
    print(f"Execution time: {execution_time_minutes} minutes")


# main function
if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run pipeline")

    # Add the command-line arguments
    parser.add_argument("experiment_name", type=str, help="Name of the experiment")
    parser.add_argument(
        "--create_df_artifact",
        action="store_true",
        help="Flag to enable creating df artifact",
    )
    parser.add_argument(
        "--open_smile_feature_count",
        type=int,
        default=10,
        help="Value for open smile feature count",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=15,
        help="Number of parallel processes to run",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="results_multivoice.csv",
        help="Path of the CSV file to save",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Check if the experiment name is provided
    if not args.experiment_name:
        parser.error("Experiment name is required.")

    # Extract the arguments
    experiment_name = args.experiment_name
    create_df_artifact = args.create_df_artifact
    open_smile_feature_count = args.open_smile_feature_count
    num_processes = args.num_processes
    save_path = args.save_path

    # Call the main function with the arguments
    main(
        experiment_name,
        open_smile_feature_count,
        create_df_artifact,
        num_processes,
        save_path,
    )
