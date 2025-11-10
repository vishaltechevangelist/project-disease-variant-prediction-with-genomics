class vector_db():
    def __init__(self, client, host, api_key, collection_name, point_obj):
        self.client = client
        self.api_key = api_key
        self.host = host
        self.collection_name = collection_name
        self.points = point_obj
        self.vector_db = client(host=self.host, api_key=self.api_key)

    def recreate_collection(self, vector_params, dim, distance):
        collection_name = self.collection_name
        if self.client.collection_exists(collection_name):
            self.client.delete_collection()

        self.client.create_collection(collection_name=self.collection_name, 
                                      vectors_config=vector_params(size=dim, distance=distance)
                                      )
        
    def create_point_struct(self, id, insert_vector, payload):
        return self.points(id=int(id), vector=insert_vector, payload=payload)
        
    def upsert(self, id, insert_vector, payload):
        points = []
        points.append(self.create_point_struct(id, insert_vector, payload))
        return self.client.upsert(collection_name=self.collection_name, points=points)
    
    def search(self, query_vector, limit, with_payload):
        result = self.client.search(self.collection_name, query_vector, limit=limit, with_payload=True)
        for data in result:
            print(data)