mkdir street-scene
cd street-scene
wget https://www.merl.com/pub/streetscene/Train.tgz
wget https://www.merl.com/pub/streetscene/Test.tgz
tar -zxvf Train.tgz
tar -zxvf Test.tgz
python streetscene.py
cd ..