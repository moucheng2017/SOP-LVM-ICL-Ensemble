if [ $# -eq 0 ]; then
    echo "Error: You need to specify which file to download. 'full', 'gold', or 'debug'"
    exit 1
fi

case "$1" in 
    "full")
        LINK=https://zenodo.org/records/12671568/files/demos.zip
        FILENAME=demos.zip
        DIRNAME=demos
        ;;
    "gold")
        LINK=https://zenodo.org/records/12671568/files/gold_demos.zip
        FILENAME=gold_demos.zip
        DIRNAME=gold_demos
        ;;
    "debug")
        LINK=https://zenodo.org/records/12671568/files/debug_demos.zip
        FILENAME=debug_demos.zip
        DIRNAME=debug_demos
        ;;
esac

mkdir -p data/demos && pushd data/demos
wget "$LINK"
unzip "$FILENAME" && rm "$FILENAME"
mv "$DIRNAME/*" . && rmdir "$DIRNAME" 
popd

echo "Finished downloading the wonderbread $FILENAME split and extracted it into data/demos"
exit 0
