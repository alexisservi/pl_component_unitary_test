# CREARTE AND COMPILE A PIPELINE
import os
import kfp
from kfp import dsl
from kfp import compiler
#import kfp.components as comp
from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Metrics,
    InputPath, OutputPath, )
from google_cloud_pipeline_components.v1.custom_job import create_custom_training_job_from_component
from typing import NamedTuple


#URL_READ_LINES_COMP = 'gs://ml-auto-pipelines-bucket/components-yamls/line-reader-writer/kubeflow_component_spec.yaml'
URL_READ_LINES_COMP = "https://storage.googleapis.com/ml-auto-pipelines-bucket/components-yamls/line-reader-writer/kubeflow_component_spec.yaml"


#---------------------------------------------------------------------------------------------------
@dsl.component() 
def file_writer(lines_to_write_1: int,
                out_file_1: OutputPath()) -> NamedTuple(
  'ExampleOutputs',
  [
    ('lines_to_read', int),
    ('test_string_out', str)
  ]):
    
    N_LINES_TO_WRITE = 20
    with open(out_file_1, 'w') as path_writer:
        for k in range(lines_to_write_1):
            if k == 0:
                path_writer.write("Test file writing\n")
            else:
                path_writer.write(str(k) + "\n")

    lines_to_read = 5
    from collections import namedtuple
    example_output = namedtuple('ExampleOutputs', ['lines_to_read', 'test_string_out'])
    return example_output(lines_to_read, out_file_1)


#---------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------
read_lines_comp = kfp.components.load_component_from_url(url=URL_READ_LINES_COMP)  # Passing pipeline parameter as argument to consumer op
custom_training_job_comp = create_custom_training_job_from_component(
    read_lines_comp, # lines_to_write_1=lines_to_write_1,), #file_writer,
    display_name = 'Custom Training Job -> with custom machine and GPU',
    machine_type = 'n1-standard-4', 
    accelerator_type='NVIDIA_TESLA_T4', # https://cloud.google.com/vertex-ai/docs/training/configure-compute#specifying_gpus
    accelerator_count='1'
)

print(repr(read_lines_comp))
print(repr(custom_training_job_comp))
#---------------------------------------------------------------------------------------------------
@dsl.pipeline(name='component_unitary_test_pipeline-v1', description='A pipeline components unitary test')
def component_unitary_test_pipeline():

    
    file_writer_task = file_writer(lines_to_write_1=7)

    """
    #--------------------------
    # START: Create a custom training job from component
    file_writer_task = file_writer(lines_to_write_1=lines_to_write_1)

    custom_training_job_task = custom_training_job_comp(
        input_1=file_writer_task.outputs["out_file_1"], # 
        parameter_1=file_writer_task.outputs["lines_to_read"],
        project='almacafe-ml-poc',
        location='us-central1',
    )
    # END: Create a custom training job from component
    #--------------------------
    """
    
    
    
    
    #--------------------------
    # START: Condition task excecution
    with dsl.Condition(file_reader_task.outputs["publish_model_cmd"] == "Publish model"):
        conditional_task = print_text(text=file_reader_task.outputs["publish_model_cmd"])
    # END: Condition task excecution
    #--------------------------
    
    
#------------------------------------------
# Compile pipeline
# V1 Compiler -> it works... 
compiler.Compiler().compile(
    pipeline_func=component_unitary_test_pipeline,
    package_path='component_unitary_test_pipeline.yaml', 
    #type_check=False
    )

print("PIPELINE COMPILED")
print("List directory files")
print(os.listdir())
