from abc import abstractmethod


import asyncio
import inspect
import io
import os
import traceback
from pydantic import BaseModel
import ray
from ray.serve.drivers import DAGDriver
from ray.serve.deployment_graph import InputNode
from ray.serve.handle import RayServeDeploymentHandle
from ray.serve.http_adapters import json_request
from ray.dag.vis_utils import _dag_to_dot, _get_nodes_and_edges
from transformers import AutoProcessor, BarkModel
from termcolor import colored
from ray import serve
# from bark import generate_audio, preload_models, SAMPLE_RATE
import torch
import numpy as np
import time
import json
import base64

from components.audio.force_alignment import ForceAligner
from components.etl.context import Context
from components.etl.util.connect_debug import connect_to_debug
from components.text.nlp.ner import named_entity_recognition_score
from retry import retry
from config import ENABLE_GPU_OFFLOADING, KAFKA_BOOTSTRAP_SERVERS
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer, ConsumerRecord

from src.message_broker.broker_types import AiJobStartMessage, AiJobStatusMessage, JobStatusEnum

import logging
logger = logging.getLogger("ray.serve")
# from kafka.coordinator.assignors.abstract import AbstractPartitionAssignor
# from kafka.coordinator.assignors.range import RangePartitionAssignor


def get_func_signature_kwargs(func):
    signature = inspect.signature(func)
    func_kwargs = signature.parameters
    return func_kwargs




class BaseConsumer:


    def __init__(self, task_name, group_id=None):
        # if pydev_config is not None:
        if os.getenv('PYDEV_DEBUG') == 'True':
            logger.info('connecting to pydev...')
            connect_to_debug()
        self.loop = asyncio.get_running_loop()
        logger.info('initializing test deployment')
        self.consumer = AIOKafkaConsumer(
            task_name,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            group_id=group_id or f"{task_name}-deployment-group",
            loop=self.loop,
            )
        self.healthy = True
        self.task_name = task_name
        

        
        self.loop.create_task(self.consume())

        signature = inspect.signature(self.run)
        self.func_kwargs = signature.parameters
        if 'params' in self.func_kwargs:            
            self.inputArgs = signature.parameters['params']._annotation
        logger.info('started test deployment...')
        self.custom_input_manager = False
        self.custom_output_manager = False

    # def __init_subclass__(cls) -> None:
    #     pass

    def get_func_kwargs(self, context: Context, job_start_msg: AiJobStartMessage):
        func_kwargs = {}
        # for k,v in job_start_msg.params.items():
        #     if k in self.func_kwargs:
        #         func_kwargs[k] = v
        input_args = self.inputArgs(**job_start_msg.params)
        if 'context' in self.func_kwargs:
            func_kwargs['context'] = context
        if 'params' in self.func_kwargs:
            func_kwargs['params'] = input_args
        else:
            raise Exception(f'params not in function kwargs: {self.func_kwargs}')


        return func_kwargs

    async def consume(self):
        await self.consumer.start()
        partitions = self.consumer.assignment()        
        logger.info(f"Consuming <{self.task_name}> messages...")
        try:       
            async for msg in self.consumer:
                try:                    
                    job_start_msg = AiJobStartMessage(**json.loads(msg.value))
                    logger.info(f"<{self.task_name}> Consumer got job:")
                    # print(job_start_msg)
                    context = Context(
                        task_name=self.task_name, 
                        job_id=job_start_msg.job_id, 
                        user_id=job_start_msg.user_id,
                        job_start_msg=job_start_msg,
                    )
                    
                    await context.init()

                    func_kwargs = self.get_func_kwargs(context, job_start_msg)
                    func_kwargs['params'] = await self.input_manager(context, func_kwargs['params'])
                    output = await self.run(**func_kwargs)
                    output = await self.output_manager(context, func_kwargs['params'], output)
                    await context.logger.emit(output, JobStatusEnum.SUCCESS)
                
                except Exception as e:
                    logger.exception(e)
                    await context.logger.error(e)
                finally:
                    await context.close()
        except Exception as e:
            logger.exception(e)
        finally:
            await self.consumer.stop()
            logger.info("Stopped consuming text2speech messages")
            self.healthy = False

    @abstractmethod
    async def run(self, context: Context, job_start_msg: AiJobStartMessage):
        raise NotImplementedError()
    
    async def input_manager(self, context: Context, params: BaseModel):
        return params

    async def output_manager(self, context: Context, params: BaseModel, output: any, ):
        return output


    async def check_health(self):
        if not self.healthy:
            raise RuntimeError("Kafka Consumer is broken")
