# Recommended environment
# To create a conda environment run: 
#	conda create -n <environment_name>
#
# To start this environment run: 
#	conda activate <environment_name>
#
# To end this environment run: 
#	conda deactivate
#
# To delete this environment run:
#	conda remove -n <environment_name> --all
#
# conda environment location:
#	/usr/local/anaconda3/envs/opencv

# check if in local environment
# install pandas, matplotlib

sudo apt-get -y update

sudo apt-get -y install vim

#ssh-keygen -t rsa

#sudo apt-get install curl

#cd /tmp

#curl -O https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh

#bash Anaconda3–2020.11-Linux-x86_64.sh

#source ~/.bashrc

#git clone git@github.com:trevorwiebe/radar-data.git

# conda create -n "radarenv" python=3

# conda activate radarenv

#conda install -y pytorch torchvision -c pytorch

yes | python3 -m pip install tensorflow[and-cuda]

yes | pip3 install pandas matplotlib scikit-learn imageio