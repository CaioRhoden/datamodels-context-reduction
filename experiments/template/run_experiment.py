
import argparse
from experiments.utils.DatamodelsExperiment import DatamodelsExperiment
import json

def main(folder_name: str, k: int, folder_path: str, config_file: str, step: str) -> None:

    """
    Runs an experiment step.

    Args:
        folder_name: str
            The name of the experiment folder.
        folder_path: str
            The path to the experiment folder.
        config_file: str
            The path to the configuration file.
        step: str
            The step to run. Valid values are:
                - "create_scheme"
                - "create_pre_collection"
                - "create_collection"
                - "train_datamodels"
                - "evaluate"
                - "plot_results"

    Returns:
        None
    """
    config = json.load(open(config_file, "r"))
    experiment = DatamodelsExperiment(folder_name, k, folder_path, config)


    if step == "create_scheme":
        experiment.create_scheme()

        """
        TODO: Create unit test to verify the creation of all files and if the specification for the input, train, dev, set is correct
        """
        
    elif step == "create_pre_collection":
        experiment.create_pre_collection()
    elif step == "create_collection":
        experiment.create_collection()
    elif step == "train_datamodels":
        experiment.train_datamodels()
    elif step == "evaluate":
        experiment.evaluate()
    elif step == "plot_results":
        experiment.plot_results()

    else:
        raise ValueError("Invalid step")
    
    print("Experiment step finished")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_name", type=str, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--folder_path", type=str, required=True)
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--step", type=str, required=True)
    args = parser.parse_args()

    main(args.folder_name, args.k ,args.folder_path, args.config_file, args.step)