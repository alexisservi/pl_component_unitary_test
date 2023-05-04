import google.cloud.aiplatform as aip



PROJECT_ID = "almacafe-ml-poc"
PROJECT_REGION = "us-central1"
PIPELINE_ROOT_PATH = "gs://ml-auto-pipelines-bucket/pipeline-runs"
# Before initializing, make sure to set the GOOGLE_APPLICATION_CREDENTIALS
# environment variable to the file path of your service account.
aip.init(
    project=PROJECT_ID,
    location=PROJECT_REGION,
)

# Prepare the pipeline job
job = aip.PipelineJob(
    display_name="unitary-test-v1",
    template_path="component_unitary_test_pipeline.json",
    pipeline_root=PIPELINE_ROOT_PATH,
    project = PROJECT_ID,
    parameter_values={
        'project_id': PROJECT_ID
    }
)

job.submit()