from pymongo import MongoClient

mongo_client = MongoClient('mongodb://localhost:27017/')
db = mongo_client['plant_disease_db']
users_collection = db['users']
users_collection.create_index('email', unique=True)
