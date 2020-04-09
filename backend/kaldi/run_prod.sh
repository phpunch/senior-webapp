cd $1
echo $0 $1 $2 $3
. ./path.sh
. ./cmd.sh

echo "Predict audios ... "

mfccdir=mfcc
data=data
exp=exp

echo "Compute MFCC features"
compute-mfcc-feats --verbose=2 --config=conf/mfcc.conf scp:$data/wav.scp ark:- | copy-feats --compress=true ark:- ark,scp:$mfccdir/raw_mfcc.ark,$data/feats.scp || exit 1;

echo "Compute VAD"
compute-vad --config=conf/vad.conf scp:$data/feats.scp ark,scp:$mfccdir/vad.ark,$data/vad.scp || exit 1;


nnet_dir=exp/pvector_net
ver=0

# delete the old one before extracting
echo "Extract Vector"
./extract_vector.sh $nnet_dir $data $nnet_dir/pvector_prod/ vector_weights_4.hdf5 || exit 1;


vector_train_dir=$nnet_dir/pvector_train_combined
vector_adapt_dir=$nnet_dir/pvector_adapt_parliament
vector_prod_dir=$nnet_dir/pvector_prod
vector_dev_dir=$nnet_dir/pvector_dev_parliament

echo "Create Trial"
bash ./create_trial.sh $vector_dev_dir/spk_xvector.scp $vector_prod_dir/xvector.scp exp/trial_prod || exit 1;

echo "Generate Scores"
$train_cmd exp/scores/log/prod-clean-scoring.log \
   ivector-plda-scoring --normalize-length=true \
   "ivector-copy-plda --smoothing=0.0 $vector_adapt_dir/plda_adapt - |" \
   "ark:ivector-subtract-global-mean $vector_adapt_dir/mean.vec scp:$vector_dev_dir/spk_xvector.scp ark:- | transform-vec $vector_train_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
   "ark:ivector-subtract-global-mean $vector_adapt_dir/mean.vec scp:$vector_prod_dir/xvector.scp ark:- | transform-vec $vector_train_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
   "cat 'exp/trial_prod' | cut -d\  --fields=1,2 |" exp/scores/scores-prod-clean || exit 1;
