import os
import boto3
import json
import argparse
import sys
import base64
from botocore.exceptions import ClientError

parser = argparse.ArgumentParser()
parser.add_argument("instance", type=str)
parser.add_argument("runname", type=str)
parser.add_argument("region", type=str)
parser.add_argument("--datascript", type=str, default=None)
parser.add_argument("-no_training_run", action='store_true')
parser.add_argument("-no_aws_env_setup", action='store_true')
args = parser.parse_args()

INSTANCE_TYPE = args.instance
RUN_NAME = args.runname
REGION = args.region
USER_DATA_FILENAME = args.datascript
NO_TRAINING_RUN = args.no_training_run
NO_AWS_ENV_SETUP = args.no_aws_env_setup
DRY_RUN = False

print("Starting instance in region", REGION)

def key_name(region):
    return "EC2_KEY_" + region + ".pem"

def security_group_name(region):
    return "EC2_SSH_SEC_GROUP_" + region

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
    return base64.b64encode(script.encode('ascii')).decode('utf-8')

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


def get_ssh_security_group_id(client, region):
    response = client.describe_security_groups()
    for sg in response["SecurityGroups"]:
        if sg["GroupName"] == security_group_name(region):
            print("SSH SecurityGroup found: %s" % sg["GroupId"])
            return sg["GroupId"]
    return None

def create_ssh_security_group(client, region):
    response = client.describe_vpcs()
    vpc_id = response.get('Vpcs', [{}])[0].get('VpcId', '')

    try:
        response = client.create_security_group(GroupName=security_group_name(region),
                                                Description='SSH Security Group' + region,
                                                VpcId=vpc_id)
        security_group_id = response['GroupId']
        print('Security Group Created %s in vpc %s.' % (security_group_id, vpc_id))

        data = client.authorize_security_group_ingress(
            GroupId=security_group_id,
            IpPermissions=[
                {'IpProtocol': 'tcp',
                 'FromPort': 22,
                 'ToPort': 22,
                 'IpRanges': [{'CidrIp': '0.0.0.0/0'}]}
            ])
        print('Ingress Successfully Set %s' % data)
    except ClientError as e:
        print(e)
    return security_group_id

def get_dlami_info(client):
    SEARCH_STR = "Deep Learning*Ubuntu*24.0"
    images = client.describe_images(Filters=[{
        "Name": "name",
        "Values": [ SEARCH_STR ]
        }])

    if len(images["Images"]) == 0:
        print("ERROR: Found no images matching search %s" % SEARCH_STR)
        return None

    if len(images["Images"]) != 1:
        print("ERROR: Found more than one image matching search %s" % SEARCH_STR)
        return None

    image = images["Images"][0]
    return image

def get_dlami_blockdevice(ami_info):
    for b in ami_info["BlockDeviceMappings"]:
        if "Ebs" in b:
            return b["DeviceName"]
    return None

def verify_and_setup_keypair(client, region):
    response = client.describe_key_pairs()
    key_defined_on_aws = False
    key_local_path = os.environ["HOME"] + "/.ssh/" + key_name(region)
    key_local_exists = os.path.exists(key_local_path)

    for k in response["KeyPairs"]:
        if k["KeyName"] == key_name(region):
            key_defined_on_aws = True
            break

    if key_local_exists and not key_defined_on_aws:
        print("ERROR: Key not defined in AWS but available locally")
        sys.exit(1)

    if not key_local_exists:
        if key_defined_on_aws:
            print("ERROR: Key defined in AWS but not available locally")
            sys.exit(1)
        else:
            print("creating new key pair")
            response = client.create_key_pair(
                            KeyName=key_name(region)
                       )
            with open(key_local_path, "w") as file:
                file.write(response["KeyMaterial"])

    print("KeyPair for region %s set up correctly" % REGION)

ec2 = boto3.client('ec2', region_name = REGION)
print("checking for running instances...")
abort_on_running_instance(ec2)
verify_and_setup_keypair(ec2, REGION)

security_group_id = get_ssh_security_group_id(ec2, REGION)
if security_group_id is None:
    print("No SSH SecuritGroupy found for region %s" % REGION)
    security_group_id = create_ssh_security_group(ec2, REGION)


dlami_info = get_dlami_info(ec2)
if dlami_info is None:
    sys.exit(1)
print("Using AMI %s (%s)" % (dlami_info["Name"], dlami_info["ImageId"]))

device_name = get_dlami_blockdevice(dlami_info)
if device_name is None:
    print("Error: could not determine device for AMI")

response = ec2.request_spot_fleet(
    DryRun=DRY_RUN,
    SpotFleetRequestConfig={
        'AllocationStrategy': 'lowestPrice',
        'IamFleetRole': "arn:aws:iam::284165529866:role/DL-Training-Spot-Fleet-Role",
        'LaunchSpecifications': [
            {
                'SecurityGroups': [
                    {
                        'GroupId': security_group_id
                    }
                ],
                'BlockDeviceMappings': [
                    {
                        'DeviceName': device_name,
                        'Ebs': {
                            'VolumeSize': 90,
                        }
                    },
                ],
                'IamInstanceProfile': {
                    'Name': 'DL-Training'
                },
                'ImageId': dlami_info["ImageId"],
                'InstanceType': INSTANCE_TYPE,
                'KeyName': key_name(REGION),
                'Monitoring': {
                    'Enabled': False
                },
                'UserData': get_user_data(USER_DATA_FILENAME, RUN_NAME, no_aws_env_setup=NO_AWS_ENV_SETUP, no_training_run=NO_TRAINING_RUN),
                'TagSpecifications': [{
                    "ResourceType": "instance",
                    'Tags': [
                        {'Key': 'Type', 'Value': 'DL-Training'},
                        {'Key': 'RunName', 'Value': RUN_NAME}
                    ]
                }]
            },
        ],
        'TargetCapacity': 1,
        'InstanceInterruptionBehavior': 'terminate'
    }
)

print(response)
