swig -c++ -python difficultyestimator.i
g++ -fPIC -c difficultyestimator.cpp
g++ -fPIC -c difficultyestimator_wrap.cxx -I/home/jiyang/anaconda3/envs/SkillRec/include/python3.7m
g++ -shared difficultyestimator.o difficultyestimator_wrap.o -o _DifficultyEstimatorGLinux.so