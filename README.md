# Boston Data Pipeline으로 분석하기

## Boston Data 적재, Preprocessing

- sklearn을 이용하여 Boston 데이터를 획득한뒤, 학습데이터와 테스트 데이터를 분할한다.
- 기존 코드를 fairing 하는 코드도 작성한다.


```
# preprocess.ipynb

from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import os


def _preprocess_data():
    X, y = datasets.load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    np.save('x_train.npy', X_train)
    np.save('x_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    
def fairing():
    if os.getenv('FAIRING_RUNTIME', None) is None:
        from kubeflow.fairing.builders.append.append import AppendBuilder
        from kubeflow.fairing.preprocessors.converted_notebook import ConvertNotebookPreprocessor
        
        DOCKER_REGISTRY = 'ykkim77'
        base_image = 'demisto/sklearn:1.0.0.29944'
        image_name = 'boston_data_preprocessing'
        builder = AppendBuilder(
            registry=DOCKER_REGISTRY,
            image_name=image_name,
            base_image=base_image,
            push=True,
            preprocessor=ConvertNotebookPreprocessor(
                notebook_file="preprocess.ipynb"
            )
        )
                                   
        builder.build()    
 
if __name__ == '__main__':
    if os.getenv('FAIRING_RUNTIME', None) is None:
        fairing()
    
    else:
        print('Preprocessing data...')
        _preprocess_data()
```


## 모델 학습

- 모델을 학습하는 코드를 작성한뒤 fairing 한다.

```
# train.ipynb

import argparse
import joblib
import numpy as np
from sklearn.linear_model import SGDRegressor
import os

def train_model(x_train, y_train):
    x_train_data = np.load(x_train)
    y_train_data = np.load(y_train)

    model = SGDRegressor(verbose=1)
    model.fit(x_train_data, y_train_data)
    
    joblib.dump(model, 'model.pkl')
    
def fairing():
    if os.getenv('FAIRING_RUNTIME', None) is None:
        from kubeflow.fairing.builders.append.append import AppendBuilder
        from kubeflow.fairing.preprocessors.converted_notebook import ConvertNotebookPreprocessor
        
        DOCKER_REGISTRY = 'ykkim77'
        base_image = 'francoisserra/sklearn-kfp'
        image_name = 'train_model'
        builder = AppendBuilder(
            registry=DOCKER_REGISTRY,
            image_name=image_name,
            base_image=base_image,
            push=True,
            preprocessor=ConvertNotebookPreprocessor(
                notebook_file="train.ipynb"
            )
        )
                                   
        builder.build() 


if __name__ == '__main__':
    if os.getenv('FAIRING_RUNTIME', None) is None:
        from kubeflow.fairing.builders.append.append import AppendBuilder
        from kubeflow.fairing.preprocessors.converted_notebook import ConvertNotebookPreprocessor
        fairing()
    else:    
        parser = argparse.ArgumentParser()
        parser.add_argument('--x_train')
        parser.add_argument('--y_train')
        args = parser.parse_args()
        train_model(args.x_train, args.y_train)
```


## 모델 평가 코드 작성

- 학습이 완료된 모델을 평가

```
#test.ipynb

from kubernetes import client
from kfserving import KFServingClient
from kfserving import constants
from kfserving import utils
from kfserving import V1alpha2EndpointSpec
from kfserving import V1alpha2PredictorSpec
from kfserving import V1alpha2TensorflowSpec
from kfserving import V1alpha2InferenceServiceSpec
from kfserving import V1alpha2InferenceService
from kubernetes.client import V1ResourceRequirements
import os
import sys
import argparse
import logging
import time

'''
'''
class KFServing(object):
    def run(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--namespace', required=False, default='practice')
        # pvc://${PVCNAME}/dir
        parser.add_argument('--storage_uri', required=False, default='/saved_model')
        parser.add_argument('--name', required=False, default='kfserving-sample')        
        args = parser.parse_args()
        namespace = args.namespace
        serving_name =  args.name
        
        api_version = constants.KFSERVING_GROUP + '/' + constants.KFSERVING_VERSION
        default_endpoint_spec = V1alpha2EndpointSpec(
                                  predictor=V1alpha2PredictorSpec(
                                    tensorflow=V1alpha2TensorflowSpec(
                                      storage_uri=args.storage_uri,
                                      resources=V1ResourceRequirements(
                                          requests={'cpu':'100m','memory':'1Gi'},
                                          limits={'cpu':'100m', 'memory':'1Gi'}))))
        isvc = V1alpha2InferenceService(api_version=api_version,
                                  kind=constants.KFSERVING_KIND,
                                  metadata=client.V1ObjectMeta(
                                      name=serving_name, namespace=namespace),
                                  spec=V1alpha2InferenceServiceSpec(default=default_endpoint_spec))        
        
        KFServing = KFServingClient()
        KFServing.create(isvc)
        print('waiting 5 sec for Creating InferenceService')
        time.sleep(5)
        
        KFServing.get(serving_name, namespace=namespace, watch=True, timeout_seconds=300)
        
if __name__ == '__main__':
    if os.getenv('FAIRING_RUNTIME', None) is None:
        from kubeflow.fairing.builders.append.append import AppendBuilder
        from kubeflow.fairing.preprocessors.converted_notebook import \
            ConvertNotebookPreprocessor

        DOCKER_REGISTRY = 'ykkim77'
        base_image = 'brightfly/kubeflow-kfserving:latest'
        image_name = 'kfserving'

        builder = AppendBuilder(
            registry=DOCKER_REGISTRY,
            image_name=image_name,
            base_image=base_image,
            push=True,
            preprocessor=ConvertNotebookPreprocessor(
                notebook_file="serving.ipynb"
            )
        )
        builder.build()
    else:
        serving = KFServing()
        serving.run()
```



## 모델 배포

- 모델 배포하는 코드 작성 (실제 배포가 아니라, 파이프라인 실습 편의를 위해 임의로 작성)

```
# deploy.ipynb
import argparse
import os


def deploy_model(model_path):
    print(f'deploying model {model_path}...')

def fairing():
    if os.getenv('FAIRING_RUNTIME', None) is None:
        from kubeflow.fairing.builders.append.append import AppendBuilder
        from kubeflow.fairing.preprocessors.converted_notebook import ConvertNotebookPreprocessor
        
        DOCKER_REGISTRY = 'ykkim77'
        base_image = 'francoisserra/sklearn-kfp'
        image_name = 'deploy_model'
        builder = AppendBuilder(
            registry=DOCKER_REGISTRY,
            image_name=image_name,
            base_image=base_image,
            push=True,
            preprocessor=ConvertNotebookPreprocessor(
                notebook_file="deploy.ipynb"
            )
        )
                                   
        builder.build()   
    
if __name__ == '__main__':
    if os.getenv('FAIRING_RUNTIME', None) is None:
        from kubeflow.fairing.builders.append.append import AppendBuilder
        from kubeflow.fairing.preprocessors.converted_notebook import ConvertNotebookPreprocessor
        fairing()    
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--model')
        args = parser.parse_args()
        deploy_model(args.model)
```

## 파이프라인 코드 작성

- 전체 분석을 파이프라인화 하는 코드 작성

```
import kfp
from kfp import dsl
import kfp.onprem as onprem
import kfp.components as comp


def preprocess_op():

    return dsl.ContainerOp(
        name='Preprocess Data',
        image='ykkim77/boston_data_preprocessing:DCEE0DDF',
        command=['python', '/app/preprocess.py'],
        arguments=[],
        file_outputs={
            'x_train': '/app/x_train.npy',
            'x_test': '/app/x_test.npy',
            'y_train': '/app/y_train.npy',
            'y_test': '/app/y_test.npy',
        }
    )
    
def train_op(x_train, y_train):

    return dsl.ContainerOp(
        name='Train Model',
        image='ykkim77/train_model:EA1DC103',
        command=['python', '/app/train.py'],
        arguments=[
            '--x_train', x_train,
            '--y_train', y_train
        ],
        file_outputs={
            'model': '/app/model.pkl'
        }
    )

def test_op(x_test, y_test, model):

    return dsl.ContainerOp(
        name='Test Model',
        image='ykkim77/test_model:7DCCD54B',
        command=['python', '/app/test.py'],
        arguments=[
            '--x_test', x_test,
            '--y_test', y_test,
            '--model', model
        ],
        file_outputs={
            'mean_squared_error': '/app/output.txt'
        }
    )

def deploy_model_op(model):

    return dsl.ContainerOp(
        name='Deploy Model',
        image='ykkim77/deploy_model:1C9C44B6',
        command=['python', '/app/deploy_model.py'],
        arguments=[
            '--model', model
        ]
    )


@dsl.pipeline(
   name='Boston Housing Pipeline',
   description='An example pipeline that trains and logs a regression model.'
)
def boston_pipeline():
    _preprocess_op = preprocess_op()
    
    _train_op = train_op(
        dsl.InputArgumentPath(_preprocess_op.outputs['x_train']),
        dsl.InputArgumentPath(_preprocess_op.outputs['y_train'])
    ).after(_preprocess_op)

    _test_op = test_op(
        dsl.InputArgumentPath(_preprocess_op.outputs['x_test']),
        dsl.InputArgumentPath(_preprocess_op.outputs['y_test']),
        dsl.InputArgumentPath(_train_op.outputs['model'])
    ).after(_train_op)

    deploy_model_op(
        dsl.InputArgumentPath(_train_op.outputs['model'])
    ).after(_test_op)
    
if __name__ == "__main__":
    import kfp.compiler as compiler
    
    kfp.compiler.Compiler().compile(boston_pipeline, 'boston.pipeline.tar.gz')

```
