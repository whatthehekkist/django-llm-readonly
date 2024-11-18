import re
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from pymongo import MongoClient
from bson.objectid import ObjectId
from config import settings
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np


# from asgiref.sync import sync_to_async
# import asyncio
# from .apps import mongo_client


### llm related refs
# https://quillbot.com/summarize (well-working web app from quillbot)
# https://github.com/huggingface/transformers/issues/1791

# mongoDB connection
client = MongoClient(settings.MONGO_URI)
# urls_collection = client['ytt']['urls']  # [db][collection]
db = client['ytt']
urls_collection = db['urls']
print("connected to DB...")

# global variables
MAX_INPUT_LENGTH = 1024
LANG_CODE = None

# summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")  # working best so far
# summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6") # 2nd
# summarizer = pipeline("summarization", model="t5-base") # 3rd
# summarizer = pipeline("summarization", model="google/pegasus-xsum") # 3rd
# summarizer = pipeline("summarization", model="google/flan-t5-large") # 3rd
# summarizer = pipeline("summarization", model="allenai/longformer-base-4096") # not the best for summarization, for classification

# paraphraser (for now, all is just ...)
paraphraser = pipeline("text2text-generation", model="facebook/bart-large")  # working best so far
# paraphraser = pipeline("text2text-generation", model="t5-base") # working at least
# paraphraser = pipeline("text2text-generation", model="google/pegasus-large") # not working, error
# paraphraser = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws") # not working, error


# Load Zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# download stopwords from nltk
nltk.download('stopwords')


""" renderer summarizer/recommendations.html """
def recommendations_view(request):
    return render(request, 'summarizer/recommendations.html')


""" 
function that recommends similar contextual documents based on keywords of a target document
"""
@csrf_exempt
def recommend_documents(request):
    if request.method == 'POST':
        doc_id = request.POST.get('docId')
        if not doc_id:
            return JsonResponse({'error': 'No document ID provided.'}, status=400)

        # find the document where _id == doc_id
        document = urls_collection.find_one({"_id": ObjectId(doc_id)})
        if not document:
            return JsonResponse({'error': 'Document not found.'}, status=404)

        # get keywords from document
        keywords = []
        if "captionTracks" in document:
            for track in document["captionTracks"]:
                if 'keywords' in track:
                    keywords.append(track['keywords'])
        print("keywords: ", keywords)

        # early-return if no keywords found
        if not keywords:
            return JsonResponse({'error': 'No keywords found in the document.'}, status=404)

        # find any documents whose languageCode in the array captionTracks starts with "en" from mongoDB collection
        documents = urls_collection.find({
            "captionTracks.languageCode": {"$regex": "^en"}
        })
        all_keywords = []
        document_ids = []
        video_ids = []
        titles = []

        # add all keywords in captionTracks.keywords of documents to all_keywords
        for document in documents:
            doc_video_id = document.get('videoId', '')
            doc_title = document.get('title', '')
            for track in document['captionTracks']:
                if 'keywords' in track:
                    all_keywords.append(track['keywords'])
                    document_ids.append(str(document['_id']))
                    titles.append(doc_title)
                    video_ids.append(doc_video_id)
        # print("all_keywords: ", all_keywords)
        # keywords_join = ', '.join(keywords)

        # TF-IDF-vectorize to analyze keyword similarity
        vectorizer = TfidfVectorizer()
        # X = vectorizer.fit_transform(all_keywords + [keywords_join])
        X = vectorizer.fit_transform(all_keywords + keywords)

        # reduce dimensionality by SVD
        svd = TruncatedSVD(n_components=1)
        svd.fit(X)
        keyword_vector = X[-1, :].toarray()

        # calc cosine similarity
        cosine_similarities = []
        for i in range(len(document_ids)):
            doc_vector = X[i, :].toarray()
            sim = np.dot(keyword_vector, doc_vector.T) / (np.linalg.norm(keyword_vector) * np.linalg.norm(doc_vector))
            cosine_similarities.append(sim[0][0])

        # get the most similar documents (keywords) to the input document (keywords)
        similar_documents = sorted(zip(document_ids, video_ids, titles, all_keywords, cosine_similarities), key=lambda x: x[4],
                                   reverse=True)

        # return top 3 similarities
        top_documents = [(doc[0], doc[1], doc[2], doc[3]) for doc in similar_documents if doc[4] > 0]
        return JsonResponse({'documents': top_documents[:4]})

    return JsonResponse({'error': 'Invalid request method.'}, status=405)


""" function that extracts meaningful word as label from input | With no classification process """
def generate_labels(request):
    if request.method == 'POST':
        text = request.POST.get('str')
        if text:
            # print(text)

            # meaningless list in En
            stop_words = list(stopwords.words('english'))

            # TF-IDF vectorization
            # vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=10)
            vectorizer = TfidfVectorizer(stop_words=stop_words)  # filter stop_words
            X = vectorizer.fit_transform([text])  # calc TF and IDF

            # extract keywords and pick meaningful keywords
            feature_names = vectorizer.get_feature_names_out()
            tokens = [feature_names[i] for i in X.nonzero()[1]]
            labels = list(set(tokens))
            print("len(labels): ", len(labels))
            print("labels: ", labels)
            print("labels[:10]: ", labels[:10])
            return JsonResponse({'labels': labels[:10]})
        else:
            return JsonResponse({'error': 'No text for labeling provided.'}, status=400)

    return JsonResponse({'error': 'Invalid request method.'}, status=400)


""" [In Use] function that extracts meaningful word as label from input """
def generate_labels_from_tfidf(text):
    # meaningless list in En
    stop_words = list(stopwords.words('english'))

    # TF-IDF vectorization
    # vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=10)
    vectorizer = TfidfVectorizer(stop_words=stop_words)  # filter stop_words
    X = vectorizer.fit_transform([text])  # calc TF and IDF

    # extract keywords and pick meaningful keywords
    feature_names = vectorizer.get_feature_names_out()
    tokens = [feature_names[i] for i in X.nonzero()[1]]
    print("important_tokens len: ", len(tokens))
    print("important_tokens: ", tokens)

    return list(set(tokens))  # return without dup



""" function that classifies labels from request. """
@csrf_exempt
def classify_labels(request):
    if request.method == 'POST':
        text_to_classify = request.POST.get('str')
        if text_to_classify:
            # print(summary)

            # Dynamically create potential labels based on the content or context
            subject_labels = generate_labels_from_tfidf(text_to_classify)  # defines context-based labels

            # Classify the text using generated labels
            result = classifier(text_to_classify, subject_labels)
            print("len(result['labels']): ", len(result['labels']))
            print("result['labels']: ", result['labels'])
            print("result['labels'][:10]: ", result['labels'][:10])
            return JsonResponse({'labels': result['labels']})
        else:
            return JsonResponse({'error': 'No string for classification provided.'}, status=400)

    return JsonResponse({'error': 'Invalid request method.'}, status=400)


""" function to collect docs having the field captionTracks.summary """
@csrf_exempt
def get_ids_with_summary(request):
    # collect _id where the field captionTracks.summary exists
    ids_with_summary = []

    documents = urls_collection.find({"captionTracks.summary": {"$exists": True}})
    for document in documents:
        ids_with_summary.append(str(document['_id']))

    print(ids_with_summary)
    return JsonResponse({'ids_with_summary': ids_with_summary})


""" [TESTING PURPOSE ONLY] function that returns paraphrased text from input summarization """
def paraphrase_summary(request):
    if request.method == 'POST':
        summary = request.POST.get('summary')

        if summary:
            # paraphrased = paraphraser(f"paraphrase: {summary}", max_length=100, num_return_sequences=1)
            paraphrased = paraphraser(summary, max_length=100, num_return_sequences=1)
            paraphrased_summary = paraphrased[0]['generated_text']
            return JsonResponse({'paraphrased_summary': paraphrased_summary})
        else:
            return JsonResponse({'error': 'No summary provided.'}, status=400)

    return JsonResponse({'error': 'Invalid request method.'}, status=400)


""" function that saves document ids to the field captionTracks.recommend in a doc """
@csrf_exempt
@require_POST  # decorator that forces POST req job only
def save_recommends(request):
    doc_id = request.POST.get('docId')
    recommends = []
    doc_ids = request.POST.get('docIds')
    video_ids = request.POST.get('videoIds')
    titles = request.POST.get('titles')

    # store docIds, videoIds, titles into recommends
    if doc_ids and video_ids and titles:
        for doc, video, title in zip(doc_ids.split(','), video_ids.split(','), titles.split('+++')):
            recommends.append({
                "docId": doc.strip(),
                "videoId": video.strip(),
                "title": title.strip()
            })

    if doc_id and recommends:
        try:
            # find the first matching doc whose captionTracks.recommend is any one of english language code.
            document = urls_collection.find_one(
                {"_id": ObjectId(doc_id), "captionTracks.languageCode": {"$regex": r"^en(-|$)"}},
                {"captionTracks.$": 1}
            )
            if document and document.get("captionTracks"):
                current_recommend = document["captionTracks"][0].get("recommends")
                print("current_recommend before update: ", current_recommend)

                # the field recommend in the target document in mongoDB doesn't exist or has no value
                if current_recommend is None:
                    # if current_recommend is none, create and update the field with the value recommend
                    result = urls_collection.update_one(
                        {"_id": ObjectId(doc_id), "captionTracks.languageCode": {"$regex": r"^en(-|$)"}},
                        {"$set": {"captionTracks.$.recommends": recommends}}
                    )
                    if result.modified_count > 0:
                        print("[success] document _id recommendations saved to DB")
                        return JsonResponse({'status': 'success'})
                    else:
                        return JsonResponse({'status': 'error', 'message': 'No update made.'}, status=404)
                else:
                    # if current_recommend is not none, inform the client with warning
                    return JsonResponse(
                        {'status': 'warning', 'message': 'Existing document _id recommendations found. Please confirm update.'}, status=409)
            else:
                return JsonResponse({'status': 'error', 'message': 'Document not found.'}, status=404)

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Invalid data'}, status=400)


""" function that overwrites keywords to the field captionTracks.keywords in a doc """
@csrf_exempt
def confirm_recommend_update(request):
    if request.method == 'POST':
        doc_id = request.POST.get('docId')
        recommends = []
        doc_ids = request.POST.get('docIds')
        video_ids = request.POST.get('videoIds')
        titles = request.POST.get('titles')

        print("doc_id: ", doc_id)

        # Check if input parameters are valid
        if not (doc_ids and video_ids and titles):
            return JsonResponse({'status': 'error', 'message': 'Invalid input data.'}, status=400)

        # Store docIds, videoIds, titles into recommends
        for doc, video, title in zip(doc_ids.split(','), video_ids.split(','), titles.split('+++')):
            recommends.append({
                "docId": doc.strip(),
                "videoId": video.strip(),
                "title": title.strip()
            })

        try:
            # Ensure doc_id is valid ObjectId
            object_id = ObjectId(doc_id)
            result = urls_collection.update_one(
                {"_id": object_id, "captionTracks.languageCode": {"$regex": r"^en(-|$)"}},
                {"$set": {"captionTracks.$.recommends": recommends}}
            )
            print(f'Update Result: {result}')

            if result.modified_count > 0:
                return JsonResponse({'status': 'success', 'message': 'Recommends updated successfully.'})
            elif result.matched_count > 0:
                return JsonResponse({
                    'status': 'info',
                    'message': 'No update made: The conditions matched, but the data is already up to date.'
                })
            else:
                print("No documents matched the query or no changes made.")
                return JsonResponse({'status': 'info', 'message': 'No update made; document may not exist or no changes to apply.'}, status=404)

        except Exception as e:
            print(f'Error occurred: {str(e)}')
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Invalid request method.'}, status=405)



""" function that saves keywords to the field captionTracks.keywords in a doc """
@csrf_exempt
@require_POST  # decorator that forces POST req job only
def save_keywords(request):
    global LANG_CODE  # LANG_CODE is re-initialized in the function summarize_text() first always
    doc_id = request.POST.get('docId')
    keywords = request.POST.get('keywords')

    if doc_id and keywords is not None:
        try:
            # find the first matching doc whose captionTracks.languageCode is any one of english language code.
            document = urls_collection.find_one(
                # {"_id": ObjectId(doc_id), "captionTracks.languageCode": "en"},
                {"_id": ObjectId(doc_id), "captionTracks.languageCode": LANG_CODE},
                {"captionTracks.$": 1}
            )

            if document and document.get("captionTracks"):
                current_keywords = document["captionTracks"][0].get("keywords")
                print("current_keywords before update: ", current_keywords)

                # the field keywords in the target document in mongoDB doesn't exist or has no value
                if current_keywords is None:
                    # if current_keywords is none, create and update the field with the value keywords
                    result = urls_collection.update_one(
                        {"_id": ObjectId(doc_id), "captionTracks.languageCode": LANG_CODE},
                        {"$set": {"captionTracks.$.keywords": keywords}}
                    )
                    if result.modified_count > 0:
                        print("[success] keywords saved to DB")
                        return JsonResponse({'status': 'success'})
                    else:
                        return JsonResponse({'status': 'error', 'message': 'No update made.'}, status=404)
                else:
                    # if current_keywords is not none, inform the client with warning
                    return JsonResponse(
                        {'status': 'warning', 'message': 'Existing keywords found. Please confirm update.'}, status=409)
            else:
                return JsonResponse({'status': 'error', 'message': 'Document not found.'}, status=404)

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Invalid data'}, status=400)


""" function that overwrites keywords to the field captionTracks.keywords in a doc """
def confirm_keyword_update(request):
    if request.method == 'POST':
        global LANG_CODE  # LANG_CODE is re-initialized in the function summarize_text() first always
        doc_id = request.POST.get('docId')
        keywords = request.POST.get('keywords')

        result = urls_collection.update_one(
            {"_id": ObjectId(doc_id), "captionTracks.languageCode": LANG_CODE},
            {"$set": {"captionTracks.$.keywords": keywords}}
        )
        if result.modified_count > 0:
            return JsonResponse({'status': 'success', 'message': 'keywords updated successfully.'})
        else:
            return JsonResponse({'status': 'error', 'message': 'No update made.'}, status=404)

    return JsonResponse({'status': 'error', 'message': 'Invalid request method.'}, status=405)



""" function that saves a summarization to the field captionTracks.summary in a doc """
@csrf_exempt
@require_POST  # decorator that forces POST req job only
def save_summary(request):
    global LANG_CODE  # LANG_CODE is re-initialized in the function summarize_text() first always
    doc_id = request.POST.get('docId')
    summary = request.POST.get('summary')

    if doc_id and summary is not None:
        try:
            # find the first matching doc whose captionTracks.languageCode is any one of english language code.
            # (cuz it has the field summary too)
            document = urls_collection.find_one(
                # {"_id": ObjectId(doc_id), "captionTracks.languageCode": "en"},
                {"_id": ObjectId(doc_id), "captionTracks.languageCode": LANG_CODE},
                {"captionTracks.$": 1}
            )

            if document and document.get("captionTracks"):
                # current script is the first matching script in English
                current_summary = document["captionTracks"][0].get("summary")
                # current_summary = next((track for track in document["captionTracks"] if track["languageCode"] == "en"),{}).get("summary")
                # print("current_summary: ", current_summary)

                if current_summary is None:
                    # if summary is null, update the field with the value summary
                    result = urls_collection.update_one(
                        # {"_id": ObjectId(doc_id), "captionTracks.languageCode": "en"},
                        {"_id": ObjectId(doc_id), "captionTracks.languageCode": LANG_CODE},
                        {"$set": {"captionTracks.$.summary": summary}}
                    )
                    if result.modified_count > 0:
                        print("current_summary: ", current_summary)
                        return JsonResponse({'status': 'success'})
                    else:
                        return JsonResponse({'status': 'error', 'message': 'No update made.'}, status=404)
                else:
                    return JsonResponse(
                        {'status': 'warning', 'message': 'Existing summary found. Please confirm update.'}, status=409)
            else:
                # if summary is not null, inform the client with warning
                return JsonResponse({'status': 'error', 'message': 'Document not found.'}, status=404)

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Invalid data'}, status=400)


""" function that overwrites summarization to the field captionTracks.summary in a doc """
def confirm_summary_update(request):
    if request.method == 'POST':
        global LANG_CODE  # LANG_CODE is re-initialized in the function summarize_text() first always
        doc_id = request.POST.get('docId')
        summary = request.POST.get('summary')

        result = urls_collection.update_one(
            {"_id": ObjectId(doc_id), "captionTracks.languageCode": LANG_CODE},
            {"$set": {"captionTracks.$.summary": summary}}
        )
        if result.modified_count > 0:
            return JsonResponse({'status': 'success', 'message': 'Summary updated successfully.'})
        else:
            return JsonResponse({'status': 'error', 'message': 'No update made.'}, status=404)

    return JsonResponse({'status': 'error', 'message': 'Invalid request method.'}, status=405)


""" function that splits into word the input texts to summarize """
def split_text(text):
    words = text.split()  # by space
    current_chunk = []
    current_length = 0
    chunks = []

    for word in words:
        word_length = len(word) + 1  # + a space
        if current_length + word_length > MAX_INPUT_LENGTH:  # > 1024
            chunks.append(" ".join(current_chunk))  # concat with " "
            current_chunk = [word]  # look at current words[word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length

    # append the rest to chunks on finishing the for loop
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


""" function that generates summarization """
### @REMARK: currently, in reusing the last session data on the same request with docId in order to prevent heavy process over mongoDB and transformer
### @TODO: it should be ideal to reuse all session data on the same reqeusts unless the program terminates.
@csrf_exempt
def summarize_text(request):
    global LANG_CODE
    script = ""
    summary = ""

    if request.method == 'POST':
        current_doc_id = request.POST.get('docId', '')
        previous_doc_id = request.session.get('previous_request', {}).get('docId')

        # existing docId in prev session
        if previous_doc_id == current_doc_id:
            if 'summarize' not in request.POST and 'script' in request.session and request.session['script']:
                print("script in session: ", request.session['script'])
                return render(request, 'summarizer/index.html', {
                    'script': request.session['script'],
                    # 'summary': ""
                })
            elif 'summary' in request.session and request.session['summary']:
                print("summary in session: ", request.session['summary'])
                # if summary exisits in session, return script와 summary
                return render(request, 'summarizer/index.html', {
                    'script': request.session['script'],
                    'summary': request.session['summary']
                })
        # new docId
        else:
            print("New docId detected:", current_doc_id)
            request.session['script'] = ""
            request.session['summary'] = ""
            request.session['previous_request'] = {}  # flush prev session


        # using docId in request body, fetch a script from a mongoDB collection
        # user clicked the button "get-script" in the form
        if 'docId' in request.POST:
            doc_id = request.POST.get('docId', '')
            # request.session['doc_id'] = doc_id

            if doc_id:
                try:
                    # find _id
                    document = urls_collection.find_one({"_id": ObjectId(doc_id)})
                    # print("document: ", document)

                    if document is None:
                        script = "document does not exist."
                    elif "captionTracks" not in document or not isinstance(document["captionTracks"], list):
                        script = "the array or field captionTracks does not exist."
                    elif len(document["captionTracks"]) == 0:
                        script = "the array captionTracks is empty."
                    else:
                        # now, captionTracks in the document is valid
                        selected_script = None
                        for track in document["captionTracks"]:
                            # find the 1st matching languageCode starting with en or en-
                            # if track.get("languageCode") == "en":
                            if re.match(r"^en(-|$)", track.get("languageCode", "")):
                                LANG_CODE = track.get("languageCode")  # init the global variable ANG_CODE here
                                selected_script = track.get("script")
                                break

                        if selected_script:
                            script = selected_script
                        else:
                            script = "no script found with the condition."

                        # print("script:", script)
                except Exception as e:
                    script = f"MongoDB query error: {str(e)}"
            request.session['script'] = script

        # user clicked the button "summarize" in the form
        if 'summarize' in request.POST:
            # get content in <textarea name='text'>
            # the reason not just using script above is cuz user could modify content in the textarea
            text_to_summarize = request.POST.get('text', '').strip()
            if text_to_summarize:
                try:
                    # print("text_to_summarize:", text_to_summarize)

                    # text_to_summarize is too long to summarize so split it by 1024 chars
                    if len(text_to_summarize) > MAX_INPUT_LENGTH:
                        chunks = split_text(text_to_summarize)
                        summaries = []

                        for chunk in chunks:
                            result = summarizer(chunk, max_length=100, min_length=30, do_sample=False)
                            if result:
                                summaries.append(result[0]['summary_text'])
                                summary = "\n\n".join(summaries)  # concat summaries

                    # text_to_summarize < 1024, so OK to get it summarized
                    else:
                        result = summarizer(text_to_summarize, max_length=100, min_length=30, do_sample=False)
                        if len(result) > 0:
                            summary = result[0]['summary_text']

                except Exception as e:
                    summary = f"error while summarizing script: {str(e)}"

            request.session['summary'] = summary

        # store current session into request.session['previous_request']
        request.session['previous_request'] = {
            'docId': request.POST.get('docId'),
            'text': request.POST.get('text')
        }

    return render(request, 'summarizer/index.html', {
        'script': request.session.get('script', script),
        'summary': request.session.get('summary', summary)
    })










