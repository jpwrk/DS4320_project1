# DS 4320 Project 1: Minimizing False Positives in Credit Card Fraud Detection
Executive summary:
Jia Park
cqb3tc
DOI: 
Press Release: 
Data: https://myuva-my.sharepoint.com/:f:/g/personal/cqb3tc_virginia_edu/IgDP25kYkhXAToyIUUZcS3tuARwprBDZEEW4XlIUfnmt56o?e=KWiiKT
Pipeline:
License:

## Problem Definition
Initial general problem: Detecting credit card fraud
Refined problem statement: How can credit card fraud detection models be improved to reduce false positives (legitimate transactions incorrectly flagged as fraudulent) while still accurately identifying actual fraud?
  
Rationale: Most fraud detection research focuses on catching as much fraud as possible, but this often comes at the cost of flagging too many legitimate transactions. In practice, this creates a frustrating problem for both banks and customers who expect their transactions to go through without interruption. By narrowing the focus to false positive reduction specifically, the project targets something meaningful that simple existing models tend to overlook.

Motivation: Credit card fraud affects millions of people every year and costs the financial industry billions of dollars annually. While catching fraud is clearly important, overly aggressive detection systems create their own set of problems, including declined purchases, locked accounts, and frustrated customers. This project was motivated by the idea that a strong fraud detection system should not only catch fraud accurately but also avoid unnecessarily disrupting the everyday transactions of legitimate customers.

Smarter Fraud Detection: Catching criminals without punishing customers
- https://myuva-my.sharepoint.com/:t:/g/personal/cqb3tc_virginia_edu/IQCnFZzNFqqNQb0UtI-_0r9pAc1-qGJMkEiwJCKgjRGC6-k?e=XJztNo

## Domain Exposition
Terminology:
|---|
Credit Card Fraud: Unauthorized use of a credit card to make purchases or withdraw funds.
|---|
Fraud Detection System: A system that monitors financial transactions and identifies suspicious or fraudulent activity.
|---|
False Positive: A legitimate transaction that is incorrectly flagged as fraudulent.
|---|
Transaction: A financial activity such as a purchase, withdrawal, or transfer made using a credit card.

Domain: This project belongs to the domain of financial technology (FinTech) and machine learning–based fraud detection. Financial institutions process millions of credit card transactions every day, making it impossible to manually monitor each one for fraudulent activity. As a result, banks and payment companies rely on automated fraud detection systems that analyze transaction data and identify suspicious patterns. These systems often use machine learning models trained on historical transaction data to predict whether a transaction is fraudulent.

Background reading folder: https://myuva-my.sharepoint.com/:f:/g/personal/cqb3tc_virginia_edu/IgCNfrvNnv7RTqshIpOKvqMNATQwMfeQ6U5zlT_hMIvaz6o?e=SoGWmS


Table:
| Title | Description | Link |
|-------|-------------|------|
|   False Positives in Credit Card Fraud Detection: Measurement and Mitigation
    |      Propose a new method for 
assessing the cost of false positives and evaluate several 
state-of-the-art fraud detection classifiers using this 
method.      |    https://myuva-my.sharepoint.com/:b:/g/personal/cqb3tc_virginia_edu/IQAINZ3zbyM2S4wY_2zdqDnTAaegpeUyhl1truLWT_7n7ks  |
| The Hidden Cost of Fraud: An Instance-Dependent
Cost-Sensitive Approach for Positive and Unlabeled Learning |   This work introduces a novel technique that integrates PU
learning and the instance-dependent cost-sensitive framework: PU-CSBoost. PU-CSBoost
can directly minimize financial loss through an instance-dependent cost measure that also
incorporates the misclassification cost due to hidden fraudsters   | https://myuva-my.sharepoint.com/:b:/g/personal/cqb3tc_virginia_edu/IQD_Xr-KkDfaSoo8wO3jKM9XAeErWJZTzXG3EFMv2qP4Knk?e=efxjDw |
 Solving the false positives problem in fraud prediction
using automated feature engineering| 
  |   In this paper, we present an automated feature engineering based approach to dramatically reduce false positives in fraud prediction. |   https://myuva-my.sharepoint.com/:b:/g/personal/cqb3tc_virginia_edu/IQBaaJMgZOzxR4Opg3pIR2kCARkYXB2gY6csKX_3GkLaBdw   |
| Reduce card fraud and costs while improving 
the cardholder experience  |   A visual article describing the negative effects of false positives upon users, and how to mitigate this.  |  https://myuva-my.sharepoint.com/:b:/g/personal/cqb3tc_virginia_edu/IQDW_QiezwzZRJa0plV0q8_4AdKrUvlncIrfCCX1nIqqdUg?e=22LRxe |
| Reducing False Positives in Credit Card Fraud Detection 
through Cost Sensitive Learning Models
  |      Explores the application of cost-sensitive learning models to effectively reduce false 
positives while maintaining high fraud detection accuracy.  | https://myuva-my.sharepoint.com/:b:/g/personal/cqb3tc_virginia_edu/IQDAdIXW4rweQZCXMTgf3N5_AVoMXICif5CTbSJYmxljAb0?e=KfyyYs |


## Data Creation


## Metadata

