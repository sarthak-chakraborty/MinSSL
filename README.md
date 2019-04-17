# MinSSL
Minimally Supervised Semi-Supervised Learning

## Objective
 In semi-supervised learning, the biggest issue is how to choose training samples that will capture the distribution of the data of the entire class without extensive manual supervision. Since, unlabeled data is cheap and easily available; one can use the unlabelled data to enhance the weak models that one builds from minimal training data. Therefore, the key challenge is to choose few data points from a big set that will be as good as the fully supervised model. In this work, we plan to use three recent technologies.  
 * A *self-attention based RNN* to project text documents into semantic vectors.  
 * Find *topology adaptive hyper-cubes* to detect the class boundaries.  
 * Use *determinantal point process* to select diverse hyper-cubes and then select representative samples for training.  
 
 
 ### Self-Attention based RNN
The main objective to use a self attention based recurrent neural network is to transform a natural language sentence to a semantic vector that can represent the embedding of the sentence. The work in this project is based on the paper "[A Structured Self-Attentive Sentence Embedding](https://arxiv.org/abs/1703.03130)", which was published in ICLR 2017.  
 `Self Attention LSTM` contains the code for the implementation of the same. To run the code, use a `python3` environment and have the necessary libraries used in the code downloaded. Run using `python classification.py` and the embedding obtained for the train and test data are stored in the corresponding folder in a 'csv' format. (P.S. Refer to [Structured-Self-Attention](https://github.com/kaushalshetty/Structured-Self-Attention) for more details regarding the implementations.  
 ![Self Attention based LSTM for Semantic Vector](https://user-images.githubusercontent.com/166852/33136258-ccc5bc08-cf72-11e7-8ddd-368e4a85a0a8.png)
 **Hyperparameters**:  
 * Length of semantic vector = 300
 * Number of epochs = 30  
 
 
### Topology Adaptive Hyper-cubes
  Initial implementation uses a Decision Tree based classification algorithm to classify the semantic vectors into various categories. Hyper-cubes are then generated for various features (here, 300) that covers the vector space.  
  `Hypercubes` contains the code for the implementation of the same. Run the code in a `python2` environment using `python Hypercubes.py`; the output is saved in `out.txt` in text format. Run `python clean.py` to generate the results in 'csv' format; results are stored in 'HyperCool.csv'. Number of hyper-cubes generated are 1173.  
![Hypercubes](https://user-images.githubusercontent.com/23696812/56296144-8eef4e00-614b-11e9-80ce-7fbecc150437.png)
  **Hyperparameters**:
  * Depth of Decision Tree = 15
  
  
  ### Determinantal Point Process
  Determinantal Point Process chooses a set of diverse hypercubes from the ones obtained in the previous section. It takes the input csv and algorithmically chooses the set of hypercubes that maximises the principal minor(Read [Determinantal point processes for machine learning](https://arxiv.org/pdf/1207.6083.pdf) for more information). Code for this section is present in the `Hypercubes` folder. Results of the hypercubes are stored in `index_dpp_hypercubes.npy`. Algorithm chooses 150 diverse hypercubes from the set. (P.S. Algorithm needs to be feeded with the number of hypercubes it needs to be chosen).  
  ![DPP](https://user-images.githubusercontent.com/23696812/56296142-8dbe2100-614b-11e9-9eb0-05a02a9c4299.png)
  **Hyperparameters**
  * Number of hypercubes to be chosen = 150
  * Weight(`FACTOR`) gien to hypercubes of different categories = 50
  * Similarity metric between two hypercubes = Inverse Euclidean Distance  
 
 
 ## Minimized Semi-Supervise Learning
 To extract the representative samples present in the selected hypercubes, run `python extractPoints.py` in `extractPoints` folder. Labelled data points are stored in `train1.csv` and their labels in `train1labels.csv`. Unlabelled data points are stored in `otherpoints.csv`. Number of representative points for 150 hypercubes are 924. Hence there are 924 labelled datapoints and 8058 unlabelled datapoints.   
 Run a semi-supervised learning algorithm on the obtained data points using `python ssl.py` within `SSL` folder. Train accuracy and Test accuracy are reported.  
 **Hyperparameters**
 * Batch size for Unsupervised learning = 100
 
 
 Read `MinSSL.pdf` for a better understanding.  
