from pymongo.mongo_client import MongoClient
from src.core.constants import MONGO_DB_CONSTANTS

class Mongo_Client:
    """
    A class for interacting with a MongoDB database.

    Attributes:
    - uri (str): The URI of the MongoDB server.

    Methods:
    - __init__(uri): Constructor method to initialize the Mongo_Client instance with the provided URI.
    - connect_to_db(db_name, collection_name): Establishes a connection to the specified database and collection.
    - insert_data(data): Inserts the provided data into the connected collection.
    - transfert_data_to_collection(receiver_collection): Transfers data from the current collection to the specified receiver_collection.
    - update_collection(update_query, update_operation): Updates documents in the connected collection based on the provided query and operation.
    - update_collection_with_fun(update_query, operation_fun): Updates documents in the connected collection using a custom function.
    - execute_aggregate_pipeline_operation(pipeline, operation_fun): Executes an aggregation pipeline operation on the connected collection and applies a custom function to each result.
    - get_collection_data(find_query={}, fields_to_return=None): Retrieves data from the connected collection based on the provided query and fields to return.
    """
    def __init__(self, uri):
        """Constructor method to initialize the Mongo_Client instance with the provided URI."""
        self.uri = uri

    def connect_to_db(self, db_name, collection_name):
        """Establishes a connection to the specified database and collection."""
        self.client = MongoClient(self.uri)
        self.collection = self.client[db_name][collection_name]

    def insert_data(self, data):
        """Inserts the provided data into the connected collection."""
        self.collection.insert_many(data)

    def transfert_data_to_collection(self, receiver_collection):
        """Transfers data from the current collection to the specified receiver_collection."""
        for data in self.collection.find():
            data[MONGO_DB_CONSTANTS.RAW_ID_FIELD] = data.pop(MONGO_DB_CONSTANTS.ID_FIELD)
            receiver_collection.insert_one(data)

    def update_collection(self, update_query, update_operation):
        """Updates documents in the connected collection based on the provided query and operation."""
        self.collection.update_many(update_query, update_operation, upsert=False)

    def update_collection_with_fun(self, update_query, operation_fun):
        """Updates documents in the connected collection using a custom function."""
        query_result = self.get_collection_data(update_query)
        for entry in query_result:
            operation_fun(self.collection, entry)

    def execute_aggregate_pipeline_operation(self, pipeline, operation_fun):
        """Executes an aggregation pipeline operation on the connected collection and applies a custom function to each result."""
        aggregate_result = list(self.collection.aggregate(pipeline))
        for entry in aggregate_result:
            operation_fun(self.collection, entry)

    def get_collection_data(self, find_query={}, fields_to_return=None):
        """
        Retrieves data from the connected collection based on the provided query and fields to return.
        
        Args:
        - find_query (dict): The query to filter documents in the collection.
        - fields_to_return (dict or None): Optional fields to return in the query results.
        
        Returns:
        - Cursor: A cursor object representing the query results.
        """
        if fields_to_return is not None:
            return self.collection.find(find_query, fields_to_return)
        return self.collection.find(find_query)


