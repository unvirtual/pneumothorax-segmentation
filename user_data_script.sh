#!/bin/bash

USER_HOME="/home/ubuntu"
ROOT_HOME="/root"
WORK_DIR="pneumothorax-segmentation"
REPO="unvirtual/pneumothorax-segmentation.git"
S3_BUCKET="s3://dl-data-and-snapshots"
S3_REGION="eu-central-1"
USER="ubuntu"
GROUP="ubuntu"

AVAIL_ZONE=$(curl http://169.254.169.254/latest/meta-data/public-hostname)
REGION=$(echo "$AVAIL_ZONE" | sed 's/[a-z]$//')

LOCAL_HOSTNAME=$(curl http://169.254.169.254/latest/meta-data/public-hostname)
if [[ ${LOCAL_HOSTNAME} =~ .*\.amazonaws\.com ]]
then
        echo "This is an EC2 instance ... OK"
else
        echo "This is not an EC2 instance, or a reverse-customized one"
	echo "EXITING"
	save_log_and_shutdown
	exit 0
fi

INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)

save_log_and_shutdown() {
	aws --region $S3_REGION s3 cp /var/log/cloud-init-output.log $S3_BUCKET/dl-training-log-$(date +%Y-%m-%d_%H:%M:%S).log --no-progress

        SPOT_FLEET_REQUEST_ID=$(aws ec2 describe-spot-instance-requests --region $REGION --filter "Name=instance-id,Values='$INSTANCE_ID'" --query "SpotInstanceRequests[].Tags[?Key=='aws:ec2spot:fleet-request-id'].Value[]" --output text)
        if [ -n "$SPOT_FLEET_REQUEST_ID" ]; then
            aws ec2 cancel-spot-fleet-requests --region "$REGION" --spot-fleet-request-ids $SPOT_FLEET_REQUEST_ID --terminate-instances
        fi
        sleep 5
	shutdown now
}



if [ -z "$RUN_DIRECTORY" ]; then
	echo "ERROR: RUN_DIRECTORY not defined. EXITING"
	save_log_and_shutdown
	exit 1
fi

if [[ $EUID -ne 0 ]]; then
	if [ -z "$NO_AWS_ENV_SETUP" ]; then
		echo "This script must be either run as root or with the flag NO_AWS_ENV_SETUP set"
		save_log_and_shutdown
		exit 1
	fi
fi

if [ -n "$NO_AWS_ENV_SETUP" ]; then
	echo "Skipping AWS EC2 Setup and data retrieval"
fi

if [ -z "$NO_AWS_ENV_SETUP" ]; then
	cd $ROOT_HOME

	aws --region $S3_REGION s3 cp $S3_BUCKET/deploy_key /root/.ssh/id_rsa --no-progress
	sudo -H -u ubuntu bash -c "source /home/ubuntu/anaconda3/bin/activate pytorch_p36; \
	pip install --upgrade pip; \
	pip install -U scikit-image; \
	pip install pydicom segmentation-models-pytorch albumentations opencv-python"
	chmod 600 /root/.ssh/id_rsa
	ssh-keyscan -H github.com >> .ssh/known_hosts

	git clone git@github.com:$REPO

	cd $WORK_DIR

	aws --region $S3_REGION s3 cp $S3_BUCKET/data-original.zip . --no-progress
	aws --region $S3_REGION s3 cp $S3_BUCKET/full_metadata_df.pkl . --no-progress
	unzip -qq data-original.zip
	echo "Unzipping data"
	rm data-original.zip

	mv $ROOT_HOME/$WORK_DIR $USER_HOME
	chown -R $USER:$GROUP $USER_HOME/$WORK_DIR

	cd $USER_HOME/$WORK_DIR
fi

sync_to_s3() {
	while true
	do
		touch ./.lastsync_to_s3
		sleep 5
		find $1 -type f -cnewer ./.lastsync_to_s3 -exec aws --region $S3_REGION s3 cp {} $2 --no-progress \;
	done
}


if [ -z "$NO_TRAINING_RUN" ]; then
	# RUN SETUP AND EXEC
	mkdir runs
	rundir_exists_on_s3=$(aws --region $S3_REGION s3 ls "$S3_BUCKET"/runs/"$RUN_DIRECTORY")
	if [ -n "$rundir_exists_on_s3" ]; then
		echo "Run dir $RUN_DIRECTORY on S3 found. Copying..."
		aws --region $S3_REGION s3 cp "$S3_BUCKET"/runs/$RUN_DIRECTORY runs --recursive --no-progress
	else
		if [ -n "$START_FROM_FINISHED_DIR" ]; then
			finished_dir_exists_on_s3=$(aws --region $S3_REGION s3 ls $S3_BUCKET/finished_runs/"$START_FROM_FINISHED_DIR")
			if [ -z "$finished_dir_exists_on_s3" ]; then
				echo "ERROR: $S3_BUCKET/finished_runs/$START_FROM_FINISHED_DIR not found"
				save_log_and_shutdown
				exit 1
			fi
			aws --region $S3_REGION s3 cp $S3_BUCKET/finished_runs/$START_FROM_FINISHED_DIR runs --recursive --no-progress
			echo "This run starts from snapshots in finished_runs/$START_FROM_FINISHED_DIR"
		else
			echo "Run dir $RUN_DIRECTORY on S3 NOT found. This is a new run"
		fi
	fi

	chown -R $USER:$GROUP $USER_HOME/$WORK_DIR/runs

	sync_to_s3 "runs" "$S3_BUCKET/runs/$RUN_DIRECTORY/" &
	sync_to_s3_pid=$!
	trap 'echo "EXITING: killing sync to S3 loop"; kill "$sync_to_s3_pid"' EXIT

	echo "Running Training ..."
	sleep 2
        sudo -H -u ubuntu bash -c "tmux new-session -d -s dl-session 'source /home/ubuntu/anaconda3/bin/activate pytorch_p36; python run_trainer.py; tmux wait-for -S finished; ' \; pipe-pane -o 'cat > /tmp/log' \; wait-for finished"

	# wait for files to be synced
	sleep 10

	aws --region $S3_REGION s3 cp runs/ $S3_BUCKET/finished_runs/$RUN_DIRECTORY/ --recursive --no-progress
	aws --region $S3_REGION s3 cp run_trainer.py  $S3_BUCKET/finished_runs/$RUN_DIRECTORY/ --no-progress

	#aws --region $S3_REGION s3 rm $S3_BUCKET/runs/$RUN_DIRECTORY/ --recursive
        save_log_and_shutdown
fi

