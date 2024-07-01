# pgds.py demonstrates the basic operations of PyMilvus, a Python SDK of Milvus.
# 1. connect to Milvus
# 2. create collection
# 3. insert data
# 4. create index
# 5. search, query, and hybrid search on entities
# 6. delete entities by PK
# 7. drop collection
import time

import numpy as np
from glob import glob
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

def list_file_in_folder(path, ext):
    return glob(path + f"/*.{ext}")

fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"
# num_entities, dim = 3000, 8
dim = 768

#################################################################################
# 1. connect to Milvus
# Add a new connection alias `default` for Milvus server in `localhost:19530`
# Actually the "default" alias is a buildin in PyMilvus.
# If the address of Milvus is the same as `localhost:19530`, you can omit all
# parameters and call the method as: `connections.connect()`.
#
# Note: the `using` parameter of the following methods is default to "default".
print(fmt.format("start connecting to Milvus"))
connections.connect("default", host="localhost", port="19530")

has = utility.has_collection("pgds")
print(f"Does collection pgds exist in Milvus: {has}")
if has:
    print(fmt.format("Drop collection `pgds`"))
    utility.drop_collection("pgds")

#################################################################################
# 2. create collection
# We're going to create a collection with 3 fields.
# +-+------------+------------+------------------+------------------------------+
# | | field name | field type | other attributes |       field description      |
# +-+------------+------------+------------------+------------------------------+
# |1|    "pk"    |   VarChar  |  is_primary=True |      "primary field"         |
# | |            |            |   auto_id=False  |                              |
# +-+------------+------------+------------------+------------------------------+
# |2|"embeddings"| FloatVector|     dim=768      |  "float vector with dim 768" |
# +-+------------+------------+------------------+------------------------------+
fields = [
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
    FieldSchema(name="filepath", dtype=DataType.VARCHAR, description="file path", is_primary=False, max_length=1000),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
]

schema = CollectionSchema(fields, "PGDS schema")

print(fmt.format("Create collection `pgds`"))
pgds = Collection("pgds", schema, consistency_level="Strong")

################################################################################
# 3. insert data
# We are going to insert 3000 rows of data into `pgds`
# Data to be inserted must be organized in fields.
#
# The insert() method returns:
# - either automatically generated primary keys by Milvus if auto_id=True in the schema;
# - or the existing primary key field from the entities if auto_id=False in the schema.

print(fmt.format("Start inserting entities"))
embedding_vectors = np.load("/DukeMTMC-reID/embeddedVector.npy")
n_vector = len(embedding_vectors)
files = list_file_in_folder("/DukeMTMC-reID/bounding_box_test", "jpg")
entities = []
for i in range(n_vector):
    entities.append({
        # provide the pk field because `auto_id` is set to False
        "pk": str(i),
        # rng.random(num_entities).tolist(),  # field random, only supports list
        "filepath" : files[i],
        "embeddings" : embedding_vectors[i][0],    # field embeddings, supports numpy.ndarray and list
    }
    )

insert_result = pgds.insert(entities)

pgds.flush()
print(f"Number of entities in Milvus: {pgds.num_entities}")  # check the num_entites

################################################################################
# 4. create index
# We are going to create an IVF_FLAT index for pgds collection.
# create_index() can only be applied to `FloatVector` and `BinaryVector` fields.
print(fmt.format("Start Creating index IVF_FLAT"))
index = {
    "index_type": "IVF_FLAT",
    "metric_type": "COSINE",
    "params": {"nlist": 128},
}

pgds.create_index("embeddings", index)

################################################################################
# 5. search, query, and hybrid search
# After data were inserted into Milvus and indexed, you can perform:
# - search based on vector similarity
# - query based on scalar filtering(boolean, int, etc.)
# - hybrid search based on vector similarity and scalar filtering.
#

# Before conducting a search or a query, you need to load the data in `pgds` into memory.
print(fmt.format("Start loading"))
pgds.load()

# -----------------------------------------------------------------------------
# search based on vector similarity
# print(fmt.format("Start searching based on vector similarity"))
# vectors_to_search = embedding_vectors[0]
# search_params = {
#     "metric_type": "COSINE",
#     "params": {},
# }

# start_time = time.time()
# result = pgds.search(vectors_to_search, "embeddings", search_params, limit=3, output_fields=["filepath"])
# end_time = time.time()

# for hits in result:
#     for hit in hits:
#         print(f"hit: {hit}, filepath field: {hit.entity.get('filepath')}")
# print(search_latency_fmt.format(end_time - start_time))

# res = client.search(
#     collection_name="test_collection", # Replace with the actual name of your collection
#     # Replace with your query vector
#     data=[[0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592]],
#     limit=5, # Max. number of search results to return
#     search_params={"metric_type": "IP", "params": {}} # Search parameters
# )

# Convert the output to a formatted JSON string
# result = json.dumps(res, indent=4)
# print(result)

# -----------------------------------------------------------------------------
# query based on scalar filtering(boolean, int, etc.)
# print(fmt.format("Start querying with `random > 0.5`"))

# start_time = time.time()
# result = pgds.query(expr="random > 0.5", output_fields=["random", "embeddings"])
# end_time = time.time()

# print(f"query result:\n-{result[0]}")
# print(search_latency_fmt.format(end_time - start_time))

# -----------------------------------------------------------------------------
# pagination
# r1 = pgds.query(expr="random > 0.5", limit=4, output_fields=["random"])
# r2 = pgds.query(expr="random > 0.5", offset=1, limit=3, output_fields=["random"])
# print(f"query pagination(limit=4):\n\t{r1}")
# print(f"query pagination(offset=1, limit=3):\n\t{r2}")


# -----------------------------------------------------------------------------
# hybrid search
# print(fmt.format("Start hybrid searching with `random > 0.5`"))

# start_time = time.time()
# result = pgds.search(vectors_to_search, "embeddings", search_params, limit=3, expr="random > 0.5", output_fields=["random"])
# end_time = time.time()

# for hits in result:
#     for hit in hits:
#         print(f"hit: {hit}, random field: {hit.entity.get('random')}")
# print(search_latency_fmt.format(end_time - start_time))

###############################################################################
# 6. delete entities by PK
# You can delete entities by their PK values using boolean expressions.
# ids = insert_result.primary_keys

# expr = f'pk in ["{ids[0]}" , "{ids[1]}"]'
# print(fmt.format(f"Start deleting with expr `{expr}`"))

# result = pgds.query(expr=expr, output_fields=["random", "embeddings"])
# print(f"query before delete by expr=`{expr}` -> result: \n-{result[0]}\n-{result[1]}\n")

# pgds.delete(expr)

# result = pgds.query(expr=expr, output_fields=["random", "embeddings"])
# print(f"query after delete by expr=`{expr}` -> result: {result}\n")


###############################################################################
# 7. drop collection
# Finally, drop the pgds collection
# print(fmt.format("Drop collection `pgds`"))
# utility.drop_collection("pgds")
