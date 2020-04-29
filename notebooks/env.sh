DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
REPO=$( cd "$DIR/.." && pwd)
PARENT=$( cd "$REPO/.." && pwd)

if [ ! -e "$DIR/env" ]; then
  virtualenv "$DIR/env" --python=python3
  source "$DIR/env/bin/activate"
  pip install -r "$DIR/requirements.txt"
else
  source "$DIR/env/bin/activate"
fi

for DEP in celsius-utils scannotate cesium3/client; do
  DEP_DIR=$PARENT/$DEP
  if [ ! -e "$DEP_DIR" ]; then
    echo "Failed to find $DEP_DIR!  Clone the appropriate repo."
  fi
  export PYTHONPATH=$DEP_DIR:$PYTHONPATH
done

PS1="(celsiustx-speedup) $PS1"

