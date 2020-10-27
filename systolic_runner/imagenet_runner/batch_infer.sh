#! /bin/bash

EMULATOR=~/esp-tools/build/bin/qemu-riscv64
RUN="${EMULATOR} ort_test -m ../quantization/models/resnet50/model_opt.onnx  -i tools/batch_out.txt  -p mxnet -x 0"


TOTAL_LINES=1000
SPLIT_SIZE=50
OUT_DIR="benchmark/resnet"
mkdir -p ${OUT_DIR}


# ----------- NO NEED TO TOUCH -----


COUNT=1
LOWER_RANGE=1
UPPER_RANGE=$((SPLIT_SIZE))
PART_FILE="${OUT_DIR}/out.${COUNT}"
NUM_PARTS=$((TOTAL_LINES/SPLIT_SIZE))

function increment_count {
  COUNT=$((COUNT + 1))
  LOWER_RANGE=$((LOWER_RANGE + SPLIT_SIZE))
  UPPER_RANGE=$((UPPER_RANGE + SPLIT_SIZE))
  PART_FILE="${OUT_DIR}/out.${COUNT}"
}

# param order:
# $1 - filename
# $2 - lower range
# $3 - upper range (can be blank to fetch the rest)
function execute {
    if [ -z "$3" ]; then # blank upper
        $RUN -r $2 > $1 2>&1 &
    else
        $RUN -r $2,$3 > $1 2>&1 &
    fi
}

# loop as long as the upper range is less than the total
while [ "$UPPER_RANGE" -lt "$TOTAL_LINES" ]
do
  echo $UPPER_RANGE
  echo "Executing on split $COUNT of $NUM_PARTS"

  # check to see if the part already exists - if so we can skip it
  if [ -f $PART_FILE ]; then
      echo "Part $COUNT already exist"
      increment_count
      continue
  fi

  #otherwise, fetch the part
  execute $PART_FILE $LOWER_RANGE $UPPER_RANGE

  # update for next loop
  increment_count
done

#Finish the remainder till the end of the file
execute $PART_FILE $LOWER_RANGE $TOTAL_LINES

wait