# comp_astro_24

Base repository for the computational astrophysics course


command: "daneel -i path_to_the_parameters.yaml -t"   , it takes the parameters from the file "path_to_the_parameters.yaml" and plots the light curve of the relative planetary transit.

command: "daneel -i path_to_the_parameters.yaml -d svm"   , it takes the parameters from the file "path_to_the_parameters.yaml" containing the path to a training and an evaluation dataset, and the kernel to use for the Support Vector Machine. The command grants access to a class to detect exoplanets using SVMs.

command: "daneel -i path_to_the_parameters.yaml -d nn"   , it takes the parameters from the file "path_to_the_parameters.yaml" containing the path to a training and an evaluation dataset, number of hidden layers, neurons, epochs, dropout layers,learning rate for Artificial Neural Networks. The command grants access to a class to detect exoplanets using NNs.


