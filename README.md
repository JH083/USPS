# Coda Env
```
conda create -n usps python=3.8 -y
conda activate usps
conda install -c conda-forge glew
```

# Install USPS
``` 
pip install -e .
cd USPS
pip install -r requirements.txt
pip install setuptools==59.5.0
cd envs/realworldrl_suite
pip install -e . 
cd ../..
```

# Possible issues
- Error: ```AttributeError: module 'setuptools._distutils' has no attribute 'version'```
     - Solution: make sure you're using setuptools 59.5.0


