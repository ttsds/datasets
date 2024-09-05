# check if python version 3.10
if [ -z "$(python3 --version | grep '3.10')" ]; then
    echo "Python version 3.10 is required"
    exit 1
fi

# check if pip is installed
if [ -z "$(pip3 --version)" ]; then
    echo "pip is required"
    exit 1
fi

# loop through directories and create virtual environment for each
for dir in $(ls -d tts/*/); do
    # remove / from directory name
    dir=${dir%?}
    # remove tts/ from directory name
    dir=${dir#tts/}
    # check if venv exists
    if [ -f "tts/.venv/$dir/bin/activate" ]; then
        echo "$dir is already setup"
        continue
    fi
    python3 -m venv ".venv/$dir"
    source "tts/.venv/$dir/bin/activate"
    cd "$dir"
    bash "setup.sh"
    cd ../..
    deactivate
done