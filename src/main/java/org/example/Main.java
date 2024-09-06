package org.example;

import com.alibaba.fastjson.JSONObject;
import io.milvus.client.MilvusServiceClient;
import io.milvus.grpc.*;
import io.milvus.param.*;
import io.milvus.param.collection.*;
import io.milvus.param.dml.InsertParam;
import io.milvus.param.dml.SearchParam;
import io.milvus.param.index.CreateIndexParam;
import io.milvus.response.SearchResultsWrapper;

import java.util.*;

public class Main {
    private static final String COLLECTION_NAME = "protobuf_test";
    private static final String ID_FIELD = "id";
    private static final String VECTOR_FIELD = "vector";
    private static final int VECTOR_DIM = 128;

    private static final Random RAN = new Random();

    private static List<Float> generateFloatVector(int dimension) {
        List<Float> vector = new ArrayList<>();
        for (int i = 0; i < dimension; ++i) {
            vector.add(RAN.nextFloat());
        }
        return vector;
    }

    private static List<List<Float>> generateFloatVectors(int count) {
        List<List<Float>> vectors = new ArrayList<>();
        for (int i = 0; i < count; ++i) {
            vectors.add(generateFloatVector(VECTOR_DIM));
        }
        return vectors;
    }


    private static void create() {
        MilvusServiceClient milvusClient = new MilvusServiceClient(ConnectParam.newBuilder()
                .withUri("http://localhost:19530")
                .build());

        List<FieldType> fieldsSchema = Arrays.asList(
                FieldType.newBuilder()
                        .withName(ID_FIELD)
                        .withDataType(DataType.Int64)
                        .withPrimaryKey(true)
                        .withAutoID(false)
                        .build(),
                FieldType.newBuilder()
                        .withName(VECTOR_FIELD)
                        .withDataType(DataType.FloatVector)
                        .withDimension(VECTOR_DIM)
                        .build()
        );

        // Create the collection with 3 fields
        milvusClient.dropCollection(DropCollectionParam.newBuilder()
                .withCollectionName(COLLECTION_NAME)
                .build());
        R<RpcStatus> ret = milvusClient.createCollection(CreateCollectionParam.newBuilder()
                .withCollectionName(COLLECTION_NAME)
                .withFieldTypes(fieldsSchema)
                .build());
        if (ret.getStatus() != R.Status.Success.getCode()) {
            throw new RuntimeException("Failed to create collection! Error: " + ret.getMessage());
        }

        // Specify an index type on the vector field.
        ret = milvusClient.createIndex(CreateIndexParam.newBuilder()
                .withCollectionName(COLLECTION_NAME)
                .withFieldName(VECTOR_FIELD)
                .withIndexType(IndexType.FLAT)
                .withMetricType(MetricType.L2)
                .build());
        if (ret.getStatus() != R.Status.Success.getCode()) {
            throw new RuntimeException("Failed to create index on vector field! Error: " + ret.getMessage());
        }

        // Call loadCollection() to enable automatically loading data into memory for searching
        milvusClient.loadCollection(LoadCollectionParam.newBuilder()
                .withCollectionName(COLLECTION_NAME)
                .build());

        System.out.println("Collection created");

        // Insert 10 records into the collection
        List<JSONObject> rows = new ArrayList<>();
        for (long i = 1L; i <= 10; ++i) {
            JSONObject row = new JSONObject();
            row.put(ID_FIELD, i);
            row.put(VECTOR_FIELD, generateFloatVector(VECTOR_DIM));
            rows.add(row);
        }

        R<MutationResult> insertRet = milvusClient.insert(InsertParam.newBuilder()
                .withCollectionName(COLLECTION_NAME)
                .withRows(rows)
                .build());
        if (insertRet.getStatus() != R.Status.Success.getCode()) {
            throw new RuntimeException("Failed to insert! Error: " + insertRet.getMessage());
        }

        // Call flush to make sure the inserted records are consumed by Milvus server, so that the records
        // be searchable immediately. Just a special action in this example.
        // In practice, you don't need to call flush() frequently.
        milvusClient.flush(FlushParam.newBuilder()
                .addCollectionName(COLLECTION_NAME)
                .build());

        System.out.println("10 entities inserted");
        milvusClient.close();
    }

    private static void test(int repeat) {
        MilvusServiceClient milvusClient = new MilvusServiceClient(ConnectParam.newBuilder()
                .withUri("http://localhost:19530")
                .build());

        for (int k = 0; k < repeat; k++) {
            System.out.println("Search:");
            R<SearchResults> searchRet = milvusClient.search(SearchParam.newBuilder()
                    .withCollectionName(COLLECTION_NAME)
                    .withTopK(30)
                    .withVectors(generateFloatVectors(1))
                    .withVectorFieldName(VECTOR_FIELD)
                    .withParams("{\"nprobe\":10}")
                    .build());
            if (searchRet.getStatus() != R.Status.Success.getCode()) {
                throw new RuntimeException("Failed to search! Error: " + searchRet.getMessage());
            }

            SearchResultsWrapper resultsWrapper = new SearchResultsWrapper(searchRet.getData().getResults());
            List<SearchResultsWrapper.IDScore> scores = resultsWrapper.getIDScore(0);
            System.out.println("The result of No.0 target vector:");
            for (SearchResultsWrapper.IDScore score : scores) {
                System.out.println(score);
            }
        }

        milvusClient.close();
    }

    public static void main(String[] args) {
        create();
        test(1);
        System.out.println("Finished");
    }
}