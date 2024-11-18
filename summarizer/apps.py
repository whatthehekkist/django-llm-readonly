from django.apps import AppConfig
# from django.conf import settings
# from pymongo import MongoClient
#
# def get_mongo_client():
#     global mongo_client
#     if mongo_client is None: 
#         mongo_client = MongoClient(settings.MONGO_URI)
#         print("successfully connected to DB")
#     return mongo_client


class SummarizerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'summarizer'

    # def ready(self):
    #     get_mongo_client() 
