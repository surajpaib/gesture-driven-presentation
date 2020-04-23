## Gesture Driven Presentation

### 1. Installation
To install the model server dependencies (Linux only!)
```
bash setup.sh
pip install -r requirements.txt
```

Replace bash with any shell 
### 2. Run the pose estimation and communication interface
```
tensorflow_model_server --model_base_path=$PWD gesture_classification_tools --rest_api_port=9000 --model_name=saved_model &

cd javascript_pose_estimation
python detector_api.py

```

### JS Package Development!

Use this only to modify javascript code. Install node and yarn first.

The js files in [javascript_pose_estimation](javascript_pose_estimation) contain the pose estimation code. Once changes are made run,
```
python prep_dist.py
```

The prep_dist file runs yarn build and also replaces the generated files with directory structure required by tornado.

Then the detector_api.py can be run and the new javascript code will be reflected.




