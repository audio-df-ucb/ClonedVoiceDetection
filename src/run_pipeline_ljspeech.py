# global packages
import os
import sys
import multiprocessing
import pandas as pd
import mlflow
import time
import argparse

# local packages
sys.path.append("/home/ubuntu/ClonedVoiceDetection/src")
import packages.ExperimentPipeline as ep


# function that runs the pipeline asynchonously
def run_pipeline(
    fake_cols,
    metadata_path,
    open_smile_feature_count,
    run_name_prefix,
    run_tags,
    models,
    create_df_artifact,
    label_type="label",
) -> None:
    # create and run pipeline object
    exp = ep.ExperimentPipeline(fake_cols, metadata_path)
    exp.generate_features(
        feature_method="all", open_smile_feature_count=open_smile_feature_count
    )
    exp.train_predict_using_models(
        run_name_prefix=run_name_prefix,
        run_tags=run_tags,
        models=models,
        create_df_artifact=create_df_artifact,
        label_type=label_type,
    )


# main function
def main(experiment_name, open_smile_feature_count, create_df_artifact, num_processes):
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

    # set the models to run
    models = ["logreg", "random_forest"]

    ####################################
    ##### start mutliprocessing ########
    ####################################

    # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=num_processes)

    # list for holding task arguments
    task_args = []

    ######################################
    ##### tasks for unlaundered data #####
    ######################################

    # mlflow tag setting
    run_tags = {"laundered": 0}
    # metadata path
    metadata_path_unlaundered = (
        "/home/ubuntu/data/wavefake_data/LJ_metadata_16000KHz.csv"
    )

    # pipeline params
    run_params = {}
    run_params["EL"] = ["ElevenLabs"]
    run_params["UD"] = ["UberDuck"]
    run_params["WF"] = ["RandWaveFake"]
    run_params["EL_UD"] = ["ElevenLabs", "UberDuck"]
    run_params["EL_UD_WF"] = ["ElevenLabs", "UberDuck", "RandWaveFake"]
    run_params["EL_UD_Fake"] = ["EL_UD_Fake"]
    run_params["Fake"] = ["Fake"]

    # get the task params for unlaundered data
    for run_name_prefix, fake_cols in run_params.items():
        # binary classifiaction  tasks
        if len(fake_cols) == 1:
            # get args tuple and append to task_args list
            args = (
                fake_cols,
                metadata_path_unlaundered,
                open_smile_feature_count,
                run_name_prefix,
                run_tags,
                models,
                create_df_artifact,
                "label",
            )
            task_args.append(args)

        # multiclass classification tasks
        else:
            # get args tuple and append to task_args list
            args = (
                fake_cols,
                metadata_path_unlaundered,
                open_smile_feature_count,
                run_name_prefix,
                run_tags,
                models,
                create_df_artifact,
                "multiclass_label",
            )
            task_args.append(args)

    ####################################
    ##### tasks for laundered data #####
    ####################################

    # mlflow tag setting
    run_tags = {"laundered": 1}
    # metadata path
    metadata_path_laundered = (
        "/home/ubuntu/data/wavefake_data/LJ_metadata_16KHz_Laundered.csv"
    )

    # pipeline params
    run_params = {}
    run_params["EL"] = ["ElevenLabs"]
    run_params["UD"] = ["UberDuck"]
    run_params["WF"] = ["RandWaveFake"]
    run_params["EL_UD"] = ["ElevenLabs", "UberDuck"]
    run_params["EL_UD_WF"] = ["ElevenLabs", "UberDuck", "RandWaveFake"]
    run_params["EL_UD_Fake"] = ["EL_UD_Fake"]
    run_params["Fake"] = ["Fake"]

    # get the task params for laundered data
    for run_name_prefix, fake_cols in run_params.items():
        # binary classifiaction  tasks
        if len(fake_cols) == 1:
            # get args tuple and append to task_args list
            args = (
                fake_cols,
                metadata_path_laundered,
                open_smile_feature_count,
                run_name_prefix,
                run_tags,
                models,
                create_df_artifact,
                "label",
            )
            task_args.append(args)

        # multiclass classification tasks
        else:
            # get args tuple and append to task_args list
            args = (
                fake_cols,
                metadata_path_laundered,
                open_smile_feature_count,
                run_name_prefix,
                run_tags,
                models,
                create_df_artifact,
                "multiclass_label",
            )
            task_args.append(args)

    ####################################
    ##### run multiprocessing ##########
    ####################################

    # run the pipeline in parallel
    pool.starmap_async(run_pipeline, task_args)

    # close the pool and wait for the work to finish
    pool.close()
    pool.join()

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
        default=20,
        help="Value for open smile feature count",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=15,
        help="Number of parallel processes to run",
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

    # Call the main function with the arguments
    main(experiment_name, open_smile_feature_count, create_df_artifact, num_processes)
