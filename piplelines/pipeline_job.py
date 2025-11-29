from azure.ai.ml import MLClient, dsl
from azure.ai.ml.entities import PipelineJob
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
    "<SUBSCRIPTION-ID>",
    "<RESOURCE-GROUP>",
    "<WORKSPACE-NAME>",
)

@dsl.pipeline()
def tumor_pipeline():
    extract = ml_client.jobs.component("extract_features")(
        input_images="azureml://datastores/workspaceblobstore/paths/lab5/raw/"
    )

    retrieve = ml_client.jobs.component("feature_retrieval")(
        input_parquet=extract.outputs.output_parquet
    )

    train = ml_client.jobs.component("train_model")(
        train_data=retrieve.outputs.train,
        test_data=retrieve.outputs.test
    )

    return {
        "model": train.outputs.model,
        "metrics": train.outputs.metrics
    }

job = tumor_pipeline()
ml_client.jobs.create_or_update(job)
