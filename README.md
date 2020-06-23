## Gesture Driven Presentation

There are two modes of running the provided software:
1. Windows Only Powerpoint Control
2. Linux Only Gesture Detection


## 1. Windows Only Powerpoint Control

### Installation

First make sure your windows system has the PowerPoint application and Python3.6 ( this specific version is needed due to compatibility issues. Download link: https://www.python.org/downloads/release/python-360/)

Next, to install all the dependencies run the following,
```
pip install -r ppt_requirements.txt
```


### Running the ppt control

Run the following in Powershell,

```
cd gesture-presentation
python start_presentation.py <FULL_PATH_TO_PPTX_FILE>

```

This should initiate the software. Please wait till the initialization is complete.



## 2. Linux Only Gesture Detection

To install the model server dependencies (Linux only!)
```
pip install -r requirements.txt
```


### Run the pose estimation and communication interface
```
cd gesture-presentation
python detector_api.py

```

### JS Package Development!

Use this only to modify javascript code. Install node and yarn first.

The js files in [gesture-presentation](gesture-presentation) contain the pose estimation code. Once changes are made run,
```
python prep_dist.py
```

The prep_dist file runs yarn build and also replaces the generated files with directory structure required by tornado.

Then the detector_api.py can be run and the new javascript code will be reflected.




