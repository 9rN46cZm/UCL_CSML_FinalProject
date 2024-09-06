RUN_DIR="run"
MASTER_PORT=12345


if ! [ -d $RUN_DIR ]; then
  mkdir $RUN_DIR
fi

jupyter nbconvert --to script --output-dir $RUN_DIR baseline.ipynb
jupyter nbconvert --to script --output-dir $RUN_DIR convert_to_longformer_1_4096.ipynb 
jupyter nbconvert --to script --output-dir $RUN_DIR convert_to_longformer_1_12800.ipynb 
jupyter nbconvert --to script --output-dir $RUN_DIR convert_to_longformer_2_4096.ipynb 
jupyter nbconvert --to script --output-dir $RUN_DIR convert_to_longformer_2_12800.ipynb 
cp ds_config.json $RUN_DIR

cd $RUN_DIR

python baseline.py
python convert_to_longformer_1_4096.py
python convert_to_longformer_1_12800.py
deepspeed --master_port $MASTER_PORT convert_to_longformer_2_4096.py
deepspeed --master_port $MASTER_PORT convert_to_longformer_2_12800.py
