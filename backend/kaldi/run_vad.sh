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
