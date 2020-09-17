# Manipulating Template Pixels for Model Adaptation of Siamese Visual Tracking

## Test on GOT-10k:

step1: change the loop_num in siamfcpp_track.py, line 372. For GOT-10k, the loop_num is 2.

step2: `python ./main/test.py --config 'experiments/siamfcpp/train/got10k/siamfcpp_googlenet-trn.yaml'`