# Duplicate Question Pair Identification

## Overview

This project predicts whether two given questions are duplicates or not, using the famous Quora Question Pairs dataset. By identifying duplicate questions, the system can help reduce redundancy in online platforms and enhance user experience.

## Application URL

[https://duplicate-question-pairs-predictor.onrender.com/](https://duplicate-question-pairs-predictor.onrender.com/)

## Dataset

The [Quora Question Pairs dataset](https://www.kaggle.com/competitions/quora-question-pairs) contains the following columns:

* **id** : Unique identifier for each row.
* **qid1** : Unique identifier for the first question.
* **qid2** : Unique identifier for the second question.
* **question1** : The first question.
* **question2** : The second question.
* **is_duplicate** : Target variable indicating whether the questions are duplicates (1) or not (0).

## Data Preprocessing

The following preprocessing steps were applied to the questions:

1. Converted text to lowercase.
2. Removed punctuation, extra spaces, and HTML tags.
3. Replaced contracted words (e.g., "we've") with their expanded forms.
4. Lemmatized the text using WordNet Lemmatizer.

## Feature Engineering

Extracted multiple features to represent the relationships between question pairs:

1. **Token-Based Features** : Focused on commonalities and differences in tokens and stop words between questions. Examples include ratios of common words and tokens relative to question lengths, and checks for identical first or last words.
2. **Length-Based Features** : Captured differences in the lengths of the questions, such as the absolute length difference and the ratio of the longest common substring to the smaller question length.
3. **Fuzzy Features** : Computed similarity scores using fuzzy string matching techniques to quantify textual similarities.
4. **TF-IDF Weighted Word2Vec** : Converted each question into 200-dimensional vectors to capture semantic information, weighted by their TF-IDF scores.

## Modeling

Experimented with various machine learning models:

* Logistic Regression
* Support Vector Classifier (SVC)
* Naive Bayes
* XGBoost
* Random Forest
* Gradient Boosting Machine (GBM)

 **Final Model** :

* Selected Random Forest as the final model due to its ***balance between accuracy and minimizing false positives.***
* False positives were prioritized to avoid merging non-duplicate questions.

## Training and Evaluation

1. Used K-Fold Cross-Validation for robust evaluation.
2. Performed hyperparameter tuning using Grid Search and Randomized Search CV.
3. Achieved an accuracy of approximately 79%.

## Deployment

The application is deployed using  **Streamlit** , allowing users to input question pairs and receive predictions on whether they are duplicates.

### URL

Access the application [here](https://chatgpt.com/c/URL-provide-your-link).

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/bhumikaxyz/duplicate-question-pairs-predictor.git
   ```
2. Navigate to the project directory:
   ```bash
   cd duplicate-question-pair
   ```
3. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   streamlit run app/app.py
   ```
2. Open the local URL provided in your terminal.
3. Input pairs of questions to get predictions on whether they are duplicates.

## Contributing

Contributions are welcome! If you have ideas to improve the project, please submit an issue or a pull request.

## License

This project is licensed under the MIT License.
