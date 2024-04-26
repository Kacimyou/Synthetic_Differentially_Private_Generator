# Differentially Private Synthetic Data

The following project aims at generating what is known as Differentially Private Synthetic Data. The synthetic data are then tested on two learning task to evaluate practical utility.

## Differential Privacy

### Sensitivity

Sensitivity measures how much the output of a function can change when a single individual's data is modified. It quantifies the impact of any one individual on the overall result.

### Privacy Budget

Privacy budget represents the maximum amount of privacy loss that can occur over multiple analyses or queries on a dataset. It is typically managed to ensure that privacy guarantees are maintained.

### Randomized Response

Randomized response is a technique used to introduce randomness into survey responses, protecting individual privacy while still allowing statistical analysis to be performed on the aggregate data.

### Applications

- **Statistical Analysis**: Differential privacy enables statistical analysis on sensitive datasets while preserving individual privacy.
- **Machine Learning**: It allows for training models on sensitive data without exposing individual records.

### Challenges

- **Utility vs. Privacy**: There is often a trade-off between the utility of the data and the level of privacy protection provided.
- **Implementation Complexity**: Implementing differential privacy mechanisms can be challenging and require expertise in both privacy and data analysis.


## Algorithms Used
  
  The algorithms used to generate this synthetic data are taken from:
- [4] L. Wasserman and S. Zhou. [A statistical framework for differential privacy](https://arxiv.org/pdf/0811.2501.pdf). Journal of the American Statistical Association, pages 375–389, 2010.

- [5] M. Boedihardjo, T. Strohmer, and R. Vershynin. [Privacy of synthetic data: A statistical framework](https://arxiv.org/pdf/2109.01748.pdf). IEEE Transactions on Information Theory, pages 520–527, 2022.

- [6] M. Boedihardjo, T. Strohmer, and R. Vershynin. [Private measures, random walks, and synthetic data](https://arxiv.org/pdf/2204.09167.pdf). arXiv preprint arXiv:2204.09167, 2022.

# Results


