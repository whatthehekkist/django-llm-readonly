from django.urls import path
from .views import *

app_name = "summarizer"
urlpatterns = [
    path('', summarize_text, name='summarizer'),
    path('confirm_summary_update/', confirm_summary_update, name='confirm_summary_update'),
    path('confirm_keyword_update/', confirm_keyword_update, name='confirm_keyword_update'),
    path('confirm_recommend_update/', confirm_recommend_update, name='confirm_recommend_update'),

    path('save_summary/', save_summary, name='save_summary'),
    path('save_keywords/', save_keywords, name='save_keywords'),
    path('save_recommends/', save_recommends, name='save_recommends'),

    path('generate_labels/', generate_labels, name='generate_labels'),
    path('classify_labels/', classify_labels, name='classify_labels'),

    path('recommendations/', recommendations_view, name='recommendations'),
    path('recommend_documents/', recommend_documents, name='recommend_documents'),

    path('paraphrase_summary/', paraphrase_summary, name='paraphrase_summary'),  # not in use
    path('ids-with-summary/', get_ids_with_summary, name='get_ids_with_summary'),  # not in use
]
