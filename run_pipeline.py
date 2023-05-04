import google.cloud.aiplatform as aip

from kfp.registry import RegistryClient
PROJECT_ID = "almacafe-ml-poc"
client = RegistryClient(host="https://us-central1-kfp.pkg.dev/{}/ml-automation-kfp-repo".format(PROJECT_ID))


#V1 Compiler -> it works...!
templateName, versionName = client.upload_pipeline( 
  file_name="component_unitary_test_pipeline.yaml",
  tags=["v1", "latest"],
  extra_headers={"description":"Component unitary test"})

"""
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
    #parameter_values={
    #    'project': PROJECT_ID
    #}
)

job.submit()
"""