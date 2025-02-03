# Team Data Kickers FC
Édouard Demotes-Mainard  
Serge Malo  
Francis Picard  
Jorawar Singh Dham  

# LICENSE
This is a fork from [sn-mvfoul](https://github.com/SoccerNet/sn-mvfoul), which under GPL 3.0.  
This fork uses the same GPL 3.0 license.  

# Windows Setup
1. Install Pre-requisites  
* [Python 3.11.9](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe)  
* [K-Lite Codec Pack Basic](https://codecguide.com/download_k-lite_codec_pack_basic.htm)  
* Download [VARS weights](https://drive.google.com/drive/folders/1N0Lv-lcpW8w34_iySc7pnlQ6eFMSDvXn?usp=share_link)  

2. Clone this repo and cd to to it
```git clone https://github.com/sergemalo/sn-mvfoul```
```cd sn-mvfoul```

3. Python venv
```c:\Python3.11.9\python.exe -m venv venv-sn-mvfouls```

4. Activate venv
```venv-sn-mvfouls\Scripts\activate```

5. Install required Python packages
```pip install -r requirements-sm.txt```

# Running VARS
You can run the VARS GUI Python to do "manual inference" on a specific set of videos.  
```cd VARS interface```
```python main.py```
Then, you can open multiple videos from either the "dataset" subfolder (there are only 4 examples) or you can use video from the dataset you can download after signing the NDA.

