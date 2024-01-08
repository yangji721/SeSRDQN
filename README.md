# SeSRDQN

SeSRDQN (**S**elf-**E**xplaining **S**kill **R**ecommendation **D**eep **Q**-**N**etwork) is designed based on our previous work ([SkillRec](https://dl.acm.org/doi/abs/10.1145/3442381.3449985)). Here, we focus on building a self-explaining and wellperforming deep reinforcement learning model to reveal the inner decision-making process of skill recommendation.

## Requirements
> python >= 3.8.18 \
> [nvidia-tensorflow](https://github.com/NVIDIA/tensorflow) == 1.15.5+nv22.08 \
> pandas >= 2.0.3 \
> scikit-learn >= 1.3.0 \
> jieba >= 0.42.1 \
> matplotlib >= 3.7.2 \
> numpy >= 1.24.3 \
> seaborn >= 0.12.2

##### Have Tested On:

| OS    | GPU | Driver Version  | CUDA Version | 
|:-:|:-:|:-:|:-:|
| Ubuntu 20.04 | NVIDIA GeForce RTX® 3090 | 515.65.01  | 11.7 | 

## Dataset Description
In this work, we conduct experiments on a real-world [job posting dataset](https://codeocean.com/capsule/7695173/tree/v2). The data is collected from a popular online recruitment website, namely [Lagou](https://www.lagou.com/). The dataset contains over 800,000 postings of various job positions across a
time span of 36 months, ranging from July 2016 to June 2019. We first applied *Jieba* to perform word segmentation on the job descriptions and filtered the n-grams with high appearing frequencies. After manually selecting and merging the words, we got 987 IT-related job skills. To reduce noise, we filtered out the job postings containing less than 10 skills and got 805,182 job postings in total.

We provide an example of job postings as follows. You can collect your own datasets and run our code.

| Date       | Number | Job Title     | Salary Range | Location | Experience Required | Skills |
|:----------:|:------:|:-------------:|:------------:|:--------:|:-------------------:|:------:|
| 2018-02-02 | 9823   | PHP Developer | 10k-20k      | Chengdu  | 1-3 years           | Website Architecture, HTTP, Operating System, Framework, Angular.js, Optimization, Performance, Programming, Yii, JavaScript, Performance Optimization, Nginx, Linux, ThinkPHP, Architecture, PHP, Apache, Network Programming |
| ...        | ...    | ...           | ...          | ...      | ...                 | ...    |



## Code Structure
- **CONFIG.py**
  - Contains configurations for the project.
  - **Important Note**: "HOME_PATH" should be properly set before running the code.
- **Sampler.py**
  - Implements code for sampling actions using various strategies.
- **Environment/**
  - Houses the code for the environment.
    - Source code written in C++.
    - Python packages for Linux, built with [SWIG](https://www.swig.org/).
- **prepare_code/**
  - Includes preprocessing codes.
    - Generating the skill graph.
    - Creating itemsets.
- **Model/**
  - Contains our primary model and baseline methods.
- **Trainers/**
  - Responsible for managing the training process.
    - Loading training data.
    - Training the models.
    - Saving the models.
    - Evaluating the models.

## Instruction for Use
After loading the dataset, you can execute our code from the root directory by running the following command in the console:
```console
python Trainers/SeSRDQNTrainer.py 
```
For additional hyperparameter settings, please refer to the details provided in the trainer files.

## Acknowledgments
Thanks to the following projects and their contributors:

- [SkillRec](https://github.com/sunyinggilly/SkillRec)
- [i-DQN](https://github.com/maraghuram/I-DQN)
- [ProSeNet](https://github.com/rgmyr/tf-ProSeNet)