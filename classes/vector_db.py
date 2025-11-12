class vector_db():
    def __init__(self, client, collection_name, point_obj):
        self.client = client
        self.collection_name = collection_name
        self.points = point_obj

    def recreate_collection(self, vector_params, dim, distance):
        collection_name = self.collection_name
        if collection_name in [c.name for c in self.client.get_collections().collections]:
            self.client.create_collection(collection_name=self.collection_name, 
                                      vectors_config=vector_params(size=dim, distance=distance)
                                      )    
        
    # def create_point_struct(self, id, insert_vector, payload):
    #     return self.points(id=int(id), vector=insert_vector, payload=payload)
        
    # def upsert(self, id, insert_vector, payload):
    #     points = []
    #     points.append(self.create_point_struct(id, insert_vector, payload))
    #     return self.client.upsert(collection_name=self.collection_name, points=points)
    
    def search(self, query_vector, limit, with_payload):
        return self.client.search(self.collection_name, query_vector, limit=limit, with_payload=True)