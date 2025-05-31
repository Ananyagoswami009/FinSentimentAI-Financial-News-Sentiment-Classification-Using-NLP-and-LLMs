The FinSentimentAI project focuses on leveraging Natural Language Processing (NLP) and Generative AI (specifically Large Language Models like BERT and GPT) to perform sentiment analysis on financial news headlines. The aim is to classify each headline into one of three sentiment categories: positive, negative, or neutral, helping financial analysts, investors, and institutions gain quick insights into public sentiment surrounding financial entities and events.

This project was developed as part of a hackathon challenge in the BFSI (Banking, Financial Services, and Insurance) domain. It addresses real-world problems where understanding news sentiment can significantly influence market behavior, investment decisions, and risk assessment.

🎯 Problem Statement
Develop a solution that can:
Predict the sentiment of financial news headlines using NLP and LLM techniques.
Classify each headline into one of three categories: Positive, Negative, or Neutral.
Additionally, answer bonus questions using LLMs and financial datasets:
Find the lowest historical share price for Nokia on days with negative sentiment.
Identify the field in which Nokia competes with Google.

🧠 Technologies & Tools Used
Programming Language: Python
Libraries:
pandas, numpy – for data handling
matplotlib, seaborn – for data visualization
scikit-learn – for ML models and evaluation
transformers (Hugging Face) – for using pre-trained models like BERT
openai or google-generativeai – for leveraging LLMs in bonus tasks
Model Used: BERT (fine-tuned for sentiment analysis)
LLM API: Google Gemini or OpenAI GPT
Data Source: train.xlsx containing financial headlines and labeled sentiment

🧪 Workflow
Data Loading & Preprocessing:
Loaded the training dataset from Excel.
Cleaned and tokenized text.
Encoded sentiment labels.
Performed exploratory data analysis (EDA) to understand class distribution.

Model Training:
Fine-tuned a pre-trained BERT model for 3-class sentiment classification.
Split data into training and validation sets.
Optimized hyperparameters to improve accuracy.

Model Evaluation:
Achieved high classification performance using metrics: Accuracy, Precision, Recall, and F1-score.
Visualized confusion matrix and class-wise predictions.

Bonus Tasks (LLM-Powered):
Used the trained model to filter headlines related to Nokia with negative sentiment.
Integrated with financial share price APIs or historical CSVs to find the lowest share price.
Queried LLM (Gemini/GPT) to determine competitive fields between Nokia and Google.

✅ Key Features
High-accuracy classification of financial news sentiment.
BERT-powered model fine-tuned on domain-specific data.
Integration of LLMs for answering business-specific analytical questions.
Bonus insights derived from combining sentiment predictions with external financial data.

📊 Use Cases
Real-time monitoring of sentiment around stocks, companies, or financial news.
Supporting trading strategies based on public perception.
Enhancing financial risk models by incorporating textual sentiment data.

📌 Future Enhancements
Build a web app to allow real-time headline input and sentiment prediction.
Extend the dataset to include multilingual financial news.
Deploy the model using Streamlit, Flask, or FastAPI for live demonstrations.

🙌 Acknowledgements:
This project was developed as part of a BFSI-focused hackathon challenge. Special thanks to the organizers and dataset providers.
