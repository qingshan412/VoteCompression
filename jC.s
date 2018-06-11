#!/bin/csh

#$ -M yding5@nd.edu	 # Email address for job notification
#$ -m abe		 # Send mail when job begins, ends and aborts
#$ -pe smp 1-12          # Specify parallel environment and legal core size
#$ -q long           # Specify queue (use ‘debug’ for development)
#$ -N vote_debug        # Specify job name

module load python cuda tensorflow    # Required modules
#source /opt/crc/c/caffe/1.0.0-rc5/1.0.0
python Ntrain.py -data_dir '../'
# examples/imagenet/create_imagenet.sh > crclog1.txt
