# School Dropout Prediction using Anomaly Detection

*Miroslav Dzhokanov, 2025*

## Abstract

**Introduction:** Numerous predictive models for student dropout have been developed to date. However, the majority of these approaches are grounded in traditional *classification* and *regression* techniques like *decision trees, random forests, support vector machines*, as well as *logistic and linear regression* models.

The application of *anomaly detection autoencoders* to dropout prediction represents a novel and largely uncharted research direction.

**Purpose:** This paper presents an *anomaly detection* approach to dropout prediction using an *autoencoder* trained on student characteristics. Emphasizing **minimal dependency on external libraries**, the implementation is designed to expose the reader to the algorithmic structure and underlying mathematical principles.

**Data:** We will be using a common students ML dataset from 2021 with 4424 rows and 37 features, available in [UC Irvine ML Repository](https://doi.org/10.24432/C5MC89).

**Results:** The known implementations (based on classifications and regressions) achieve between 70% and 95% accuracy. The autoencoder model achieve **93%** accuracy. 

**Conclusion:** Anomaly detection is a valid and effective technique for identifying potential school dropouts, and its implementation from scratch is relatively straightforward.

**Keywords:** *dropout prediction, anomaly detection, autoencoder*
