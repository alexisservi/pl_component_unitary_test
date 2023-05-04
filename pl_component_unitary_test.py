# CREARTE AND COMPILE A PIPELINE
import os
#import kfp
from kfp.v2 import dsl
#from kfp import compiler
from kfp.v2 import compiler
from kfp import components
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
import numpy as np
from sklearn.linear_model import LinearRegression

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
@dsl.component() 
def component_test():
  #------------------------------------------------------------
  # componentWrapper class definition
  # En esta clase se envuelve el componente a probar
  class componentWrapper():
    def __init__(self, component_inputs):
      self.component_inputs = component_inputs
    
    def component_function_test(self):
      #------------------------------------------------------------
      # Aqui se pone la funcion del componente
      def train_model(X_train, y_train):
          model = LinearRegression()
          model.fit(X_train, y_train)
          return model
      
      #------------------------------------------------------------
      # Aqui se llama a la funcion del component con las entradas apropiadas
      X_train = self.component_inputs["X_train"]
      y_train = self.component_inputs["y_train"]
      component_output = train_model(X_train, y_train)

      # Se retorna la salida del componente
      return component_output

  #------------------------------------------------------------
  # Codigo que hace prueba unitaria al componente
  # Se crean unas entradas apropiadas
  X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
  y_train = np.array([10, 23, 34])
  component_inputs = {"X_train": X_train, "y_train": y_train}

  # Se crea la instancia de la "envoltura" del componente
  test_component_instance = componentWrapper(component_inputs)

  # Se llama a la funcion que prueba el componente
  component_output = test_component_instance.component_function_test()

  # Se verifica que devuelve un modelo
  print("type(component_output): {}".format(type(component_output)))
  
  # En este caso la salida es un modelo, entonces se evalua el R squared
  r_squared = component_output.score(X_train, y_train)

  print("r_squared: {}".format(r_squared))

  # Si el R squared es mayor a un umbral se aprueba el componente
  if r_squared > 0.7:
    component_ok = True
  else:
    component_ok = False

  print("component_ok: {}".format(component_ok))
  return component_ok

#---------------------------------------------------------------------------------------------------
read_lines_comp = components.load_component_from_url(url=URL_READ_LINES_COMP)  # Passing pipeline parameter as argument to consumer op
custom_training_job_comp = create_custom_training_job_from_component(
    component_test, # lines_to_write_1=lines_to_write_1,), #file_writer,
    display_name = 'Component Test -',
    machine_type = 'n1-standard-4', 
    #accelerator_type='NVIDIA_TESLA_T4', # https://cloud.google.com/vertex-ai/docs/training/configure-compute#specifying_gpus
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


    custom_training_job_task = custom_training_job_comp(
        input_1=file_writer_task.outputs["out_file_1"], # 
        parameter_1=file_writer_task.outputs["lines_to_read"],
        project='almacafe-ml-poc',
        location='us-central1',
    )
    # END: Create a custom training job from component
    #--------------------------
    """
    
    
    
#------------------------------------------
# Compile pipeline
"""
# V1 Compiler -> it works... 
compiler.Compiler().compile(
    pipeline_func=component_unitary_test_pipeline,
    package_path='component_unitary_test_pipeline.yaml', 
    #type_check=False
    )
"""

# V2 Compiler -> 
compiler.Compiler().compile(
    pipeline_func=component_unitary_test_pipeline,
    package_path='component_unitary_test_pipeline.json', 
    #type_check=False
    )

print("PIPELINE COMPILED")
print("List directory files")
print(os.listdir())
