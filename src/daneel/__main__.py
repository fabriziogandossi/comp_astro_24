import datetime
import argparse
from daneel.parameters import Parameters
from daneel.detection.transit import plot_transit
from daneel.detection.svm import SVMExoplanetDetector
from daneel.detection.nn import NeuralNetworkDetector


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        dest="input_file",
        type=str,
        required=True,
        help="Input par file to pass",
   )
    
    parser.add_argument(
        "-d",
        "--detect",
        dest="detect",
        type=str,
        required=False,
        choices=["svm", "nn", "cnn"],
        help="Specify the detection method to use.",
    )

    parser.add_argument(
        "-a",
        "--atmosphere",
        dest="complete",
        required=False,
        help="Atmospheric Characterisazion from input transmission spectrum",
        action="store_true",
    )

    parser.add_argument(
        "-t",
        "--transit",
        dest="transit",  # Corrected to use "transit" as the destination
        required=False,
        help="Plot transit light curve",
        action="store_true",
    )

    args = parser.parse_args()

    """Launch Daneel"""
    start = datetime.datetime.now()
    print(f"Daneel starts at {start}")

    input_pars = Parameters(args.input_file).params

    if args.transit:
        plot_transit(input_pars)

    if args.detect:
        pass
    #if args.atmosphere:
    #   pass

    # Run detection methods
    '''if args.detect:
        if args.detect == "svm":
            kernel = input_pars.get("kernel", "linear")
            dataset_path = input_pars.get("dataset_path", "dataset.csv")
            detector = SVMExoplanetDetector(kernel=kernel, dataset_path=dataset_path)
            detector.train_and_evaluate()'''

    if args.detect:
        if args.detect == "svm":
            # Fetch dataset paths for training and evaluation
            train_dataset_path = input_pars.get("train_dataset_path", "")
            eval_dataset_path = input_pars.get("eval_dataset_path", "")
            kernel = input_pars.get("kernel", "linear")

            # Initialize and run the SVM detector
            detector = SVMExoplanetDetector(
                kernel=kernel,
                train_dataset_path=train_dataset_path,
                eval_dataset_path=eval_dataset_path,
            )
            detector.train_and_evaluate()

    if args.detect:
        if args.detect == "nn":
            nn_params = input_pars.get("nn", {})
            detector = NeuralNetworkDetector(**nn_params)
            detector.train_and_evaluate()


    finish = datetime.datetime.now()
    print(f"Daneel finishes at {finish}")


if __name__ == "__main__":
    main()
