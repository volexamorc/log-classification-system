This project introduces a hybrid system for log classification that blends three different techniques, each suited to a specific type of pattern complexity. By using multiple methods, the system stays adaptable and effective across simple, intricate, and poorly-labeled log data.

Classification Techniques Used

1. Regular Expressions (Regex):
Best for straightforward and consistent patterns, regex works well with logs that follow clear, rule-based formats.

2. Sentence Transformers with Logistic Regression:
This method is ideal for more complex patterns where labeled data is available. It turns log entries into vector representations using Sentence Transformers, then classifies them with Logistic Regression.

3. Large Language Models (LLMs):
When there’s not enough labeled data and the patterns are too complicated for traditional methods, LLMs step in. They serve as a backup or complement to the other approaches.


Setup Instructions
Install Dependencies: Make sure you have Python installed on your system. Install the required Python libraries by running the following command:

pip install -r requirements.txt

