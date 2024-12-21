import datetime
import argparse
from daneel.parameters import Parameters
from daneel.detection.transit import plot_transit
from daneel.detection.svm import SVMExoplanetDetector
from daneel.detection.nn import NeuralNetworkDetector
from daneel.detection.cnn import CNNPipeline
from daneel.detection.atmosphere import ForwardModel
from daneel.detection.atmosphere import Retrieval

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
        dest="atmosphere",
        required=False,
        choices=["model", "retrieve"],
        help="Atmospheric Characterisazion from input transmission spectrum",
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
    
    if args.atmosphere:
        if args.atmosphere=="model":
            fm = ForwardModel(args.input_file)
            fm.run()
    
    if args.atmosphere:
        if args.atmosphere=="retrieve":
            fm = Retrieval(args.input_file)
            fm.run()

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
            
    if args.detect == "cnn":
            print("Using CNN for detection...")
            pipeline = CNNPipeline(
				train_path=input_pars['train_dataset_path'],
                eval_path=input_pars['eval_dataset_path'],
				batch_size=input_pars['batch_size'],
				learning_rate=input_pars['learning_rate'],
				epochs=input_pars['epochs'],
				kernel_size=input_pars['kernel_size']
    		)
            pipeline.run()


    finish = datetime.datetime.now()
    print(f"Daneel finishes at {finish}")


if __name__ == "__main__":
    main()
