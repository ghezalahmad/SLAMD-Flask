# SLAMD-Flask

Here we developed Flask web-based SLAMD app (jupyter-based: https://github.com/BAMcvoelker/SequentialLearningApp) that uses machine learning to speed up the experimental search for suitable materials.

# Installation
## Python3
```
pip install -r requirements.txt
```
before to run the requirement file,  set up the enviroments.

## Developing
We used Python Flask framework along with ML, JQ, HTML, CSS, and extra python pacakges.
## Prerequisites

``` 
conda create -n 'your_env_name' python
conda activatate 'your_env_name'
git clone https://github.com/ghezalahmad/SLAMD-Flask.git
```

In order to run the app, cd `SLAMD-FLASK` folder and type:
``` 
python app.py
```
Go to your browser and look for port : ``` 127.0.0.1:5000```
## File Structure
```
├───datasets
│   └───.ipynb_checkpoints
├───preprocessed
├───static
│   ├───css
│   └───js
├───templates

```

# How to use the app?
The app is separated into four primary pages, which are discussed below: "Upload," "Data Info," "Preprocessing" "Design Space Explorer," "Sequenital Learning" and "Materials Discovery".

## Upload

<img width="963" alt="Capture" src="https://user-images.githubusercontent.com/1660323/155502570-54a20d89-6b47-4586-a3f8-d42e3a3aa3b3.PNG">

## Data Info

<img width="956" alt="Capture1" src="https://user-images.githubusercontent.com/1660323/155503072-c2cd6e87-3d1b-48fd-9172-c622cbe0c676.PNG">

## Preprocessing:

In this page, user can clean their dataset and select their appropriate features from dataset.
<img width="953" alt="Capture2" src="https://user-images.githubusercontent.com/1660323/155503839-7aad1970-039a-4bec-be34-3f769a1c8927.PNG">


## Design Space Explorer

<img width="802" alt="Capture4" src="https://user-images.githubusercontent.com/1660323/155503973-38b28d20-2fa3-4f6d-a187-9dd3dee0a9f5.PNG">
<img width="762" alt="Capture5" src="https://user-images.githubusercontent.com/1660323/155503984-89ecfd91-1199-41e9-9583-8cfffcc15ed3.PNG">


## Benchmarking
<img width="957" alt="Capture7" src="https://user-images.githubusercontent.com/1660323/155504156-30a9252f-0526-4f2b-9739-a7bbe8f69f95.PNG">


