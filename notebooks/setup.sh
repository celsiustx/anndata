DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
REPO=$( cd "$DIR/.." && pwd)
PARENT=$( cd "$REPO/.." && pwd)

if [ ! -e "$DIR/env" ]; then
  virtualenv "$DIR/env" --python=python3 || exit 1
fi


pip install -r "$DIR/requirements.txt" || exit 1

EXIT=1
for DEP in celsius-utils scannotate cesium3/client; do
  DEP_DIR=$PARENT/$DEP
  export PYTHONPATH=$DEP_DIR:$PYTHONPATH
  if [ ! -e "$DEP_DIR" ]; then
    echo "Failed to find $DEP_DIR!  Clone the appropriate repo!"
    EXIT=1
  else
    pip install -r $DEP_DIR/requirements.txt
  fi
done

