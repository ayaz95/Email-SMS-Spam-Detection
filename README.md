# Katch - Your Shield Against Fraud

## About 
Katch is an innovative solution that leverages the power of cloud-based Machine Learning (ML) to provide a comprehensive and dynamic defense against fraudulent activities. 
This platform is designed to detect, prevent, and mitigate various types of scams and to identify potential scammers. 


## Project Structure
- `data/`: Contains the datasets.
- `Team/`: Team photos.
- `Machine Learning Model Files` - Email_Scam_Code.ipynb, Fuzzy_Logic_Code.ipynb, and Identity_Code.ipnb
- `Main application File` - app.py
- `README.md`: This file.

## Dataset
- `Categorised_data.csv` - This file contains scammers information like Names, Age, Gender, Country, etc used to calculate the fuzzy logic matching score
- `Exposure_Rate_by_scam_type_2020-22.csv` - This file contains information related to types of scams happened from 2020-2022
- `Scam_exposure_by_age_2021-22.csv` - This file contains the information related to scam exposure by age
- `Scam_exposure_by_mode_2020-2022.csv` - This file contains information related to modes of scams like emails, sms, phone calls, etc
- `Scam_reporting_authorities_2020-22.csv` - This file contains information related to authorities who have been notified once the scam has happened like Banks, Police, etc
- `train.csv` - This file contains the text emails/sms that banks usually receive from their customers on their daily basis
- `Dating_data.csv` - This file is also contains information related to scammers who specifically have done dating scams in the past.

## Requirements
- Python 3.x
- nltk
- NumPy
- pandas
- scikit-learn
- pickle
- fuzzywuzzy

Operator can install the above pcakages from requirement.txt file

## Install required packages:
- pip install -r requirements.txt

## AI and Machine Learning Models Usage

- `Multinomial Naive Bayes classification Machine Learning models`

Katch platform is using a `Multinomial Naive Bayes classification Machine Learning model`(field of natural language processing and text classification) to detect whether emails and texts provided by the users falls under which category. 

Here's a detailed explanation of how this process works:

1. Data Collection:

Gather a dataset of emails that are already labeled as Account Enquiry, Account Update, Positive Customer Review, Negative Customer Review and Spam from Hugging face website(https://huggingface.co/datasets/consumer-finance-complaints). 
This dataset should include the email content and corresponding labels.

2. Data Preprocessing:

Clean and preprocess the text data. This may involve tasks such as:
Removing special characters and punctuation.
Converting text to lowercase to ensure uniformity.
Tokenizing the text into individual words or terms.
Removing common stop words like "and," "the," "in," etc.

3. Feature Extraction:

Convert the text data into numerical features that can be used by the Multinomial Naive Bayes classifier. Common techniques include:
Bag of Words (BoW): Create a vocabulary of all unique words in the dataset and represent each email as a vector of word counts.
Term Frequency-Inverse Document Frequency (TF-IDF): Assign a weight to each word based on its importance in the document and its frequency across the dataset.

4. Splitting the Data:

Divide the dataset into two parts: a training set and a testing set. The training set is used to train the Multinomial Naive Bayes model, while the testing set is used to evaluate its performance.

5. Model Training:

Train a Multinomial Naive Bayes classifier on the training data. The Multinomial Naive Bayes algorithm is a probabilistic classifier that works well with discrete features, such as word counts. It assumes that the features are conditionally independent, which is a simplifying assumption often made in text classification tasks.

6. Model Evaluation:

Evaluate the model's performance on the testing data using various metrics such as:
Accuracy: The proportion of correctly classified emails.
Precision: The ratio of true positive predictions to the total predicted positives.
Recall: The ratio of true positive predictions to the total actual positives.
F1-score: The harmonic mean of precision and recall.

7. Results

              precision    recall  f1-score   support						
						
           0       0.96      0.42      0.58       117						
           1       1.00      0.54      0.70        24						
           2       0.94      0.29      0.44       104						
           3       0.99      0.97      0.98        86						
           4       1.00      0.56      0.71        36						
           5       0.78      1.00      0.87       576						
						
    accuracy                           0.82       943						
   macro avg       0.94      0.63      0.72       943						
weighted avg       0.85      0.82      0.79       943						

8. Predictions:

Use the trained Multinomial Naive Bayes model to make predictions on new, unseen emails. The model assigns a probability score to each email, indicating the likelihood of it being one of the target variables. 
A threshold is set to classify emails based on these probability scores.

9. Deployment:

Deploy the trained model into a production environment using pickle library and call the trained model while testing the dataset.


## Reason to use Multinomial Naive Bayes Classifier 

Multinomial Naive Bayes is a popular choice for email spam detection because it's relatively simple, efficient, and performs well on text classification tasks. 
However, it's important to note that no model is perfect, and regular updates and monitoring are essential to maintain the accuracy of the spam detection system.


-- `Fuzzy Logic Matching Algorithm`

Our app uses FuzzyWuzzy python library. This library contains `fuzz.token_set_ratio` function which is commonly used for fuzzy string matching and similarity comparison. 
It uses a combination of several text comparison techniques to calculate a similarity score between two strings. The primary techniques involved in the token_set_ratio method include:

`Tokenization`: The input strings are tokenized into individual words or terms. Tokenization is a fundamental step in text processing, allowing the algorithm to work with individual units of text.

`Set Operations`: The algorithm performs set operations on the tokens. Specifically, it computes the intersection and union of the token sets from the two input strings.

`Levenshtein Distance (Edit Distance)`: For the common tokens (intersection), the algorithm calculates the Levenshtein distance or edit distance. This distance metric measures the minimum number of insertions, deletions, and substitutions required to transform one string into another.

`Scoring`: The algorithm assigns a similarity score based on the Levenshtein distance and the sizes of the intersection and union of the token sets. The exact scoring method used by FuzzyWuzzy may vary but often involves weighted combinations of these factors.

The token_set_ratio method, in particular, focuses on a "token set" approach, where it considers the set of tokens common to both strings and their variations. This approach is especially useful for cases where the input strings may have some differences, and you want to measure their similarity while allowing for some token variation or reordering.

Overall, fuzz.token_set_ratio provides a similarity score that quantifies the likeness of two strings, considering token overlap and character-level differences, and is often used for tasks like fuzzy string matching and deduplication. It is a versatile tool for applications where approximate matching or similarity comparison is required.


## Acknowledgements 

- Thanks to Equifax for providing this opportunity to participate in DevJam
- Kaggle for open source datasets - (https://www.kaggle.com/datasets/venky73/spam-mails-dataset/ , https://www.kaggle.com/datasets/darpan25bajaj/bank-reviewcomplaint-analysis)
- Hugging face for dataset - https://huggingface.co/datasets/consumer-finance-complaints
- Australian Bureau of Statistics - https://www.abs.gov.au/statistics/people/crime-and-justice/personal-fraud/latest-release#scams
- Cybersentinel Team Members

