#!/bin/bash

if [[ -z $1 ]]; then
    echo "preprocess.sh SESIDX"
    exit
fi

sesidx=$1
sesid=$(sed -n ${sesidx}p "${PWD}/metadata/PPMI_BIDS_complete_sub_ses.txt")
subid=$(echo $sesid | cut -d " " -f 1)
sesid=$(echo $sesid | cut -d " " -f 2)

datadir="${PWD}/bids_complete"
outdir="${PWD}/fmriprep"
logdir="${PWD}/logs/fmriprep"

fs_license=$(readlink -f ../../resources/license.txt)
# we need to separately mount a shared fsaverage directory, otherwise there is a race
# https://github.com/nipreps/fmriprep/issues/3492
fsavgdir=$(readlink -f ../../resources/fsaverage)

mkdir -p $outdir 2>/dev/null
mkdir -p $logdir 2>/dev/null

docker run --rm \
    -v "${datadir}:/data:ro" \
    -v "/tmp/datasets/ppmi/bids/:/tmp/datasets/ppmi/bids/:ro" \
    -v "${outdir}:/out" \
    -v "${fsavgdir}:/out/sourcedata/freesurfer/fsaverage:ro" \
    -v "${fs_license}:/opt/freesurfer/license.txt:ro" \
    nipreps/fmriprep:25.2.3 \
    /data /out participant \
    --participant-label $subid \
    --session-label $sesid \
    --skip_bids_validation \
    --fs-license-file /opt/freesurfer/license.txt \
    --ignore fieldmaps slicetiming sbref t2w flair fmap-jacobian \
    --output-spaces T1w MNI152NLin6Asym:res-2 \
    --cifti-output 91k \
    --nprocs 1 \
    --omp-nthreads 1 \
    --subject-anatomical-reference sessionwise \
    --stop-on-first-crash \
    2>&1 | tee -a ${logdir}/${subid}_${sesid}.txt
