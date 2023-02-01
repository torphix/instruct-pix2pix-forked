import boto3
import asyncio
import logging
import meadowrun
import numpy as np
from PIL import Image
from inference import predict
import requests
import io
from PIL import Image

inputs = {
    "input": "https://forked-server-user-images.s3.eu-west-2.amazonaws.com/example.jpg",
    "edit": "make him a cybernetic ninja",
}


print(
    asyncio.run(
        meadowrun.run_function(
            predict,
            meadowrun.AllocEC2Instance(),
            meadowrun.Resources(
                logical_cpu=1,
                memory_gb=4,
                max_eviction_rate=80,
                gpu_memory=10,
                flags=["nvidia"],
            ),
            meadowrun.Deployment.git_repo(
                "https://github.com/torphix/instruct-pix2pix-forked.git",
                interpreter=meadowrun.CondaEnvironmentYmlFile("environment.yaml"),
            ),
            args=[inputs],
        )
    )
)
