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
 **Hyperparameters**:  
 * Length of semantic vector = 300
 * Number of epochs = 30  
 
 
### Topology Adaptive Hyper-cubes
  Initial implementation uses a Decision Tree based classification algorithm to classify the semantic vectors into various categories. Hyper-cubes are then generated for various features (here, 300) that covers the vector space.  
  `Hypercubes` contains the code for the implementation of the same. Run the code in a `python2` environment using `python Hypercubes.py`. Then run `python clean.py` and the results are stored in 'HyperCool.csv'.  
  **Hyperparameters**:
  * Depth of Decision Tree = 15
  
  
  ### Determinantal Point Process
  
