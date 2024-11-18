# Text Summarization, classification, keyword similiarity calculation
- Hugging Face Transformer
  - summarizer model: summarization, facebook/bart-large-cnn
  - classification model: zero-shot-classification, facebook/bart-large-mnli
- nltk, skit-learn 
  - similarity analysis: stopwords, TfidfVectorizer, TruncatedSVD  

# Program flow
fetch text input from a target document in mongoDB<br>
➡️ summarize input <br>
➡️ extract & classify keywords <br>
➡️ calc keyword similarities between the target document and all documents <br>
➡️ insert document recommendations into the target document

# output in deployed webapp
## <a href="https://ytt.koyeb.app/" target="_blank">ytt.koyeb.app</a>

# env
- pycharm community
- django/python


# installation (venv)
```commandline
pip install django~=4.0 transformers scikit-learn torch pymongo
```

