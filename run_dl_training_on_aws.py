import boto3
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("instance", type=str)
parser.add_argument("runname", type=str)
parser.add_argument("--datascript", type=str, default=None)
parser.add_argument("-no_training_run", action='store_true')
parser.add_argument("-no_aws_env_setup", action='store_true')
args = parser.parse_args()

INSTANCE_TYPE = args.instance
RUN_NAME = args.runname
USER_DATA_FILENAME = args.datascript
NO_TRAINING_RUN = args.no_training_run
NO_AWS_ENV_SETUP = args.no_aws_env_setup
AMI_ID = "ami-012e02509f5e9b11f"
DRY_RUN = False

if args.datascript is None and (args.no_training_run or args.no_aws_env_setup):
    print("-no_training_run and -no_aws_env_setup require the --datascript filename option")
    print("Exiting...")
    sys.exit(1)

def get_user_data(filename, name, no_training_run, no_aws_env_setup):
    if filename is None:
        print("Using NO User Data Script")
        return ""

    print("Using User Data Script from", filename)
    with open(filename, "r") as file:
        script = file.readline()
        script += 'RUN_DIRECTORY="' + name + '"\n'
        if no_training_run:
            script += 'NO_TRAINING_RUN=1\n'
        if no_aws_env_setup:
            script += 'NO_AWS_ENV_SETUP=1\n'
        script += file.read()
    return script

#print(get_user_data(USER_DATA_FILENAME, RUN_NAME, no_aws_env_setup=NO_AWS_ENV_SETUP, no_training_run=NO_TRAINING_RUN))

def abort_on_running_instance(ec2):
    response = ec2.describe_instances()
    for r in response["Reservations"]:
        for i in  r["Instances"]:
           state = i["State"]["Name"]
           if state == "running" or state == "stopped":
               print("Error: Found a running/stopped instance already:")
               print (i)
               print("Aborting")
               sys.exit()

ec2 = boto3.client('ec2', region_name = "eu-central-1")
print("checking for running instances...")
abort_on_running_instance(ec2)

response = ec2.run_instances(
    MinCount = 1,
    MaxCount = 1,
    DryRun = DRY_RUN,
    BlockDeviceMappings=[
        {
            'DeviceName': '/dev/sda1',
            'Ebs': {
                'VolumeSize': 90,
            }
        },
    ],
    SecurityGroups=[
        'DL-Training-EC2-SecGroup'
    ],
    InstanceInitiatedShutdownBehavior = 'terminate',
    ImageId = AMI_ID,
    InstanceType = INSTANCE_TYPE,
    Monitoring = {
        'Enabled': False
    },
    UserData = get_user_data(USER_DATA_FILENAME, RUN_NAME, no_aws_env_setup=NO_AWS_ENV_SETUP, no_training_run=NO_TRAINING_RUN),
    IamInstanceProfile = {
        'Name': 'DL-Training'
    },
    KeyName = "AWS_EC2_Key",
    TagSpecifications = [{
        "ResourceType": "instance",
        'Tags': [
            {'Key': 'Type', 'Value': 'DL-Training'},
            {'Key': 'RunName', 'Value': RUN_NAME}
        ]
    }]
)

print(response)
