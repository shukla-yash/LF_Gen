# LF_Gen
Code Base and Technical Appendix for Curriculum Schema Transfer

The experiments were conducted using a 64-bit Linux Machine, having Intel(R) Core(TM) i9-9940X CPU @ 3.30GHz processor and 126GB RAM memory. The maximum duration for running the experiments was set at 24 hours.

Terminology for Source Code: LF - Low-Fidelity HF - High-Fidelity

To generate the automated curriculum in the LF environment, run: $ python LF_wo_Fire/curr.py

To test the automated curriculum transfer method with noisy mapping in HF: 
$ python Crafter-TurtleBot/Noisy_exp/HF_wo_Fire/test_curr.py
$ python Panda-Pick-And-Place/Panda_noise/curriculumTest.py

Running the above programs would generate a log file that stores the number of timesteps, rewards, episodes and other information from the curriculum run and the learning from scratch run. After conducting 10 trials, the results can be plotted using the plot_episode.py file.

The paper demonstrates results from 10 trails. The experiments are conducted on seeds 1-10

To test the automated params, generate the params using Crafter-TurtleBot/LF_wo_Fire/curr.py and Panda-Pick-And-Place/Panda_LF/curr.py and change the params on #47-54 of python LF_wo_Fire/test_curr.py

To test the automated curriculum in the HF environment, run: 
$ python Crafter-TurtleBotHF_wo_Fire/PG/test_curr.py
$ python Panda-Pick-And-Place/Panda_curriculum_2/curriculumTest.py
