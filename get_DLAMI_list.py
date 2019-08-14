import json
import boto3
ec2_client = boto3.client('ec2', region_name='eu-central-1') 
print(ec2_client.describe_key_pairs())
