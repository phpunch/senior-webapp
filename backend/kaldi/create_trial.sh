
echo "### CREATE TRIALS FILE ###"
echo "### at ./exp/trials    ###"

trials=$3

if [ -f $trials ]; then
    # if trials exists already
    rm $trials
fi


spk_ivecs=$1 #./exp/xvector_nnet_1a/xvectors_test-clean/spk_xvector.scp
utt_ivecs=$2 #./exp/xvector_nnet_1a/xvectors_test-clean/xvector.scp



while read utt; do
    utt=( $utt );
    utt=${utt[0]}
    uttspk=(${utt//_/ })
    while read spk; do
        spk=( $spk );
        spk=${spk[0]};
        speaker=(${spk//-/ })
        speaker=${speaker[0]}
        # echo $speaker $uttspk
        if [ "$speaker" == "$uttspk" ]; then 
            echo $spk $utt "target" >> $trials;
        else
            echo $spk $utt "nontarget" >> $trials;
        fi
    done <$spk_ivecs;
done <$utt_ivecs;
