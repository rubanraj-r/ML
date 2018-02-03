#                                                  #
# Example to get clarity in SVM for Classification #
#                                                  #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer();
cancer_data = pd.DataFrame(cancer['data'], columns = cancer['feature_names'])
print(cancer_data.head(5))