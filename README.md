# Text Summarizer
- Hugging Face Transformer
  - summarizer model: facebook bart-large-cnn

# env
- pycharm community
- django/python


# installation (venv)
```commandline
pip install django~=4.0 transformers torch pymongo
```

# TOC
- [config/urls.py](#config/urls.py)
- [summarizer/urls.py](#summarizer/urls.py)
- [summarizer/views.py](#summarizer/views.py)
- [templates/summarizer/index.html](#templates/summarizer/index.html)
- [static/css/styles.css](#static/css/styles.css)



# config/urls.py
```python
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('summarizer.urls')),
]
```

# summarizer/urls.py
```python
from django.urls import path
from .views import *

app_name = "summarizer"
urlpatterns = [
    path('', summarize_text, name='summarizer'),
    path('save_summary/', save_summary, name='save_summary'),
    path('confirm_summary_update/', confirm_summary_update, name='confirm_summary_update'),
    path('paraphrase_summary/', paraphrase_summary, name='paraphrase_summary'),  # not in use
    path('ids-with-summary/', get_ids_with_summary, name='get_ids_with_summary'),  # not in use
]
```

# summarizer/views.py
```python
import re
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST
from transformers import pipeline
from django.views.decorators.csrf import csrf_exempt
from pymongo import MongoClient
from bson.objectid import ObjectId
from config import settings

# mongoDB connection
client = MongoClient(settings.MONGO_URI)
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



### function to collect docs having the field captionTracks.summary ###
@csrf_exempt
def get_ids_with_summary(request):
    # collect _id where the field captionTracks.summary exists
    ids_with_summary = []

    documents = urls_collection.find({"captionTracks.summary": {"$exists": True}})
    for document in documents:
        ids_with_summary.append(str(document['_id']))

    print(ids_with_summary)
    return JsonResponse({'ids_with_summary': ids_with_summary})


### [TESTING PURPOSE ONLY] function that returns paraphrased text from input summarization ###
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



### function that saves a summarization to the field captionTracks.summary in a doc ###
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
                print("current_summary: ", current_summary)

                if current_summary is None:
                    # if summary is null, update the field with the value summary
                    result = urls_collection.update_one(
                        # {"_id": ObjectId(doc_id), "captionTracks.languageCode": "en"},
                        {"_id": ObjectId(doc_id), "captionTracks.languageCode": LANG_CODE},
                        {"$set": {"captionTracks.$.summary": summary}}
                    )
                    if result.modified_count > 0:
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



### function that overwrites summarization to the field captionTracks.summary in a doc ###
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


### fucntion that splits into word the input texts to summarize ###
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



"""
function that generates summarization 
@REMARK: currently, in reusing the last session data on the same request with docId in order to prevent heavy process over mongoDB and transformer
@TODO: it should be ideal to reuse all session data on the same reqeusts unless the program terminates.
"""
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
                # if summary exisits in session, return script and summary
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
```


# templates/summarizer/index.html
```html
<!DOCTYPE html>
<html lang="ko" xmlns="http://www.w3.org/1999/html">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Text Summarizer</title>

    {% load static%}
    <link rel="stylesheet" href="{% static 'css/styles.css' %}">

    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
</head>
<body>
    <div>
        <h1>Text Summarizer</h1>
        <form method="POST" id="summary-form">
            {% csrf_token %}
            <label for="docId">Document ID:</label>
            <input type="text" name="docId" id="docId" placeholder="enter a document ID (mongoDB)"
                   value="{{ request.POST.docId }}"
                   required
                   {% comment %}
                    {% if script or summary %} disabled {% else %} required {% endif %}
                   {% endcomment %}
            >
            <br>
            <textarea name="text" rows="10" cols="50"
                      placeholder="or you can enter text here">
                {% if summary %}{% else %} {{ script }} {% endif %}
            </textarea>
            <br>

            <button type="submit" name="get_script">Load Script</button>
            <button type="submit" name="summarize">Summarize</button>
        </form>
    </div>

    <div id="summary-result">
         {% if summary %}
            <h2>Summarization</h2>
            <div>
                <p>{{ summary|linebreaks }}</p>
            </div>

            <div>
                <button id="paraphrase-summary" type="button">Paraphrase Summary</button>
                <p id="paraphrased-output"></p>
            </div>

            <div>
                <button id="save-summary" type="button">Save Summary to DB</button>
            </div>

            <div>
                <h2>Source Script</h2>
                <p><i>{{ script }}</i></p>
            </div>
        {% endif %}
    </div>

    <!--<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>-->
    <script>
        $(document).ready(function() {

            // page refresh requested
            if (window.performance && window.performance.navigation.type === window.performance.navigation.TYPE_RELOAD) {

                console.log("page refresh detected...");
                $('#summary-form #docId').val('');
                $('#summary-form textarea[name="text"]').val('');
                $('#summary-result').hide();

                //window.location.href = window.location.pathname; // force page refresh
                //location.href = location.href;
            }

             // <button id="save-summary" type="button">Save Summary to DB</button>
            $('#save-summary').on('click', function() {
                let temp_summary = "";
                if ($('#paraphrased-output').is(':empty')) { // no paraphrased content exists so go for summary
                    temp_summary = "{{ summary|escapejs }}";
                    console.log("Save Summary to DB: ", temp_summary);
                } else {
                    // paraphrased content exists (TODO: the quality is too low, not good for production)
                    temp_summary = $('#paraphrased-output').text();
                    console.log("Save Summary to DB: ", temp_summary);
                }

                const summary = temp_summary;
                const docId = $('#docId').val();

                function saveSummary() {
                    $.ajax({
                        url: '{% url "summarizer:save_summary" %}',
                        type: 'POST',
                        data: {
                            'summary': summary,
                            'docId': docId,
                            'csrfmiddlewaretoken': '{{ csrf_token }}'
                        },
                        success: function(response) {
                            if (response.status === 'success') {
                                alert('Successfully saved summarization to DB');
                            }
                        },
                        error: function(xhr) {
                        // 409 Conflict
                        if (xhr.status === 409) {
                            const response = JSON.parse(xhr.responseText);
                            console.log(response.status); // warning
                            const confirmUpdate = confirm(response.message + "\nSURE to UPDATE?");
                            if (confirmUpdate) {
                                $.ajax({
                                    url: '{% url "summarizer:confirm_summary_update" %}',
                                    type: 'POST',
                                    data: {
                                        'summary': summary,
                                        'docId': docId,
                                        'csrfmiddlewaretoken': '{{ csrf_token }}'
                                    },
                                    success: function() {
                                        alert('Successfully saved (over-written) summarization to DB');
                                    },
                                    error: function(xhr, status, error) {
                                        alert('error while saving (over-writting): ' + error);
                                    }
                                });
                            }
                        } else {
                            alert('error while saving: ' + error);
                        }
                    }
                    });
                }

                // call saveSummary
                saveSummary();
            });

            // <button id="paraphrase-summary" type="button">Paraphrase Summary</button>
            $('#paraphrase-summary').on('click', function() {
                const summary = "{{ summary|escapejs }}";

                $.ajax({
                    url: '{% url "summarizer:paraphrase_summary" %}',
                    type: 'POST',
                    data: {
                        'summary': summary,
                        'csrfmiddlewaretoken': '{{ csrf_token }}'
                    },
                    success: function(response) {
                        $('#paraphrased-output').text(response.paraphrased_summary);
                        $('#paraphrased-output').show();
                    },
                    error: function(xhr, status, error) {
                        alert('error while paraphrasing: ' + error);
                    }
                });
            });

        });
    </script>
</body>
</html>
```

# static/css/styles.css
```css
body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 20px;
    background-color: #f4f4f4;
}

h1 {
    text-align: center;
    color: #333;
}

form {
    display: flex;
    flex-direction: column;
    max-width: 600px;
    margin: 0 auto;
    background: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
}

label {
    margin-bottom: 5px;
    font-weight: bold;
}

input[type="text"],
textarea {
    margin-bottom: 15px;
    padding: 10px;
    border: 1px solid #ccc;
    border-radius: 4px;
    font-size: 16px;
    width: 100%;
    box-sizing: border-box;
}

button {
    margin: 10px;
    padding: 10px;
    border: none;
    border-radius: 4px;
    background-color: #5cb85c;
    color: white;
    font-size: 16px;
    cursor: pointer;
    transition: background-color 0.3s ease;
}

button:hover {
    background-color: #4cae4c;
}

@media (max-width: 600px) {
    form {
        padding: 15px;
    }

    button {
        font-size: 14px;
    }

    input[type="text"],
    textarea {
        font-size: 14px;
    }
}

h2 {
    color: #333;
    margin-top: 20px;
}

p {
    margin: 10px 0;
}

#paraphrased-output {
    display: none;
    margin-top: 10px;
    padding: 10px;
    border: 1px solid #ccc;
    border-radius: 4px;
    background-color: #e9ecef;
}
```
