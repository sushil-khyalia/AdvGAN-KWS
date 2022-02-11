git clone https://github.com/castorini/honk.git
apt-get install python3-pyaudio
apt-get install freeglut3-dev
cd honk
./fetch_data.sh
mkdir -p ../honk_model/
cp ../honk_src/* utils/
cp ../dct_filter.npy ./
python3 -m utils.train \
--audio_preprocess_type MFCCs \
--data_folder	../data \
--gpu_no 0 \
--wanted_words yes no marvin left right \
--n_labels 7 \
--use_nesterov \
--batch_size 64 \
--dev_every 1 \
--n_epochs 20 \
--output_file "../honk_model/honk_kws_model.pt"
cd ..
