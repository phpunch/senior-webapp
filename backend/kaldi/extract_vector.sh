#!/bin/bash

# Copyright     2018 Hossein Zeinali (Brno University of Technology)

# Apache 2.0.

# This script extracts embeddings (called "xvectors" here) from a set of
# utterances, given features and a trained DNN.  The purpose of this script
# is analogous to sid/extract_ivectors.sh: it creates archives of
# vectors that are used in speaker recognition.  Like ivectors, xvectors can
# be used in PLDA or a similar backend for scoring.

# Begin configuration section.
nj=40
cmd="run.pl"

chunk_size=-1     # The chunk size over which the embedding is extracted.
                  # If left unspecified, it uses the max_chunk_size in the nnet directory.
use_gpu=false
stage=0

echo "${0} $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
  echo "Usage: ${0} <nnet-dir> <data> <xvector-dir> <model-file>"
  echo " e.g.: ${0} exp/xvector_nnet data/train exp/xvectors_train 1_vector0.h5"
  echo "main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --use-gpu <bool|false>                           # If true, use GPU."
  echo "  --nj <n|30>                                      # Number of jobs"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --chunk-size <n|-1>                              # If provided, extracts embeddings with specified"
  echo "                                                   # chunk size, and averages to produce final embedding"
fi

srcdir=$1
data=$2
dir=$3
model_file=$4

for f in ${srcdir}/models/${model_file} ${data}/feats.scp ${data}/vad.scp ; do # ${srcdir}/min_chunk_size ${srcdir}/max_chunk_size
  [ ! -f ${f} ] && echo "No such file $f" && exit 1;
done

min_chunk_size=`cat ${srcdir}/min_chunk_size 2>/dev/null`
max_chunk_size=`cat ${srcdir}/max_chunk_size 2>/dev/null`

model_dir=${srcdir}/models

if [ ${chunk_size} -le 0 ]; then
  chunk_size=${max_chunk_size}
fi

if [ ${max_chunk_size} -lt ${chunk_size} ]; then
  echo "${0}: specified chunk size of ${chunk_size} is larger than the maximum chunk size, ${max_chunk_size}" && exit 1;
fi

mkdir -p ${dir}/log

# utils/split_data.sh --per-utt ${data} ${nj} ################ -> is done? in data/test-clean/split40 ถ้าไม่ทำจะไม่มี vad.scp
echo "${0}: extracting xvectors for ${data}"
# Set up the features
feature_rspecifier="apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:${data}/feats.scp ark:- | select-voiced-frames ark:- scp:${data}/vad.scp ark:- |"

if [ ${stage} -le 0 ]; then
  echo "${0}: extracting xvectors from nnet"
  # if ${use_gpu}; then
  #   for g in $(seq ${nj}); do
  #     ${cmd} --gpu 1 "${dir}/log/extract.${g}.log" \
  #       /DriveE/linux/python_script/model/extract_embedding.py \
  #         --use-gpu=no --min-chunk-size=${min_chunk_size} --chunk-size=${chunk_size} \
  #         --feature-rspecifier="`echo ${feature_rspecifier} | sed s/JOB/${g}/g`" \
  #         --vector-wspecifier="| copy-vector ark:- ark,scp:${dir}/xvector.${g}.ark,${dir}/xvector.${g}.scp" \
  #         --model-dir="${model_dir}" || exit 1 &
  #   done
  #   wait
  # else
  ## Warning if .tmp is exist, this code will ignore
  i=1
    echo "extract vector ..."
    ${cmd} "${dir}/log/extract.log" \
      local/tf/extract_embedding.py \
        --use-gpu=no --min-chunk-size=${min_chunk_size} --chunk-size=${chunk_size} \
        --feature-rspecifier="${feature_rspecifier}" \
        --vector-wspecifier="| copy-vector ark:- ark,scp:${dir}/xvector.ark,${dir}/xvector.scp" \
        --model-dir="${model_dir}" \
        --model-file="${model_file}" || exit 1;
  # fi
fi

# if [ ${stage} -le 2 ]; then
#   # Average the utterance-level xvectors to get speaker-level xvectors.
#   echo "${0}: computing mean of xvectors for each speaker"
#   run.pl ${dir}/log/speaker_mean.log \
#     ivector-mean ark:${data}/spk2utt scp:${dir}/xvector.scp \
#     ark,scp:${dir}/spk_xvector.ark,${dir}/spk_xvector.scp ark,t:${dir}/num_utts.ark || exit 1;
# fi



### ./new_extract_vector.sh model/ data/test-clean/ model/pvector_test/
### ./new_extract_vector.sh model/ data/train_combined/ model/pvector_train_combined/