# INSTRUCTIONS

These are the steps to get your COCO dataset image loader up and running :-

1. Clone this repo recursively.
2. Additionally, if you wish to clone the dependecies manually, do step 3
3. run
```bash
cd src/
git submodule add https://github.com/cocodataset/cocoapi
```
4. Build the submodule by running the following
```bash
cd src/cocoapi/PythonAPI/
make
```
5. Additionally symlink the pycocotools in the cocoapi's PythonAPI directory into src/
This can be done by the following
```bash
cd src/
ln -s ./cocoapi/PythonAPI/pycocotools ./
``` 
