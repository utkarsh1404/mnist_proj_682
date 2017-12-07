python mnist_model_dcgan.py --num_epochs=50 --loss_func=1 --run=ce_0 > run_ce_0.txt
python mnist_model_dcgan.py --num_epochs=50 --lr=0.002 --loss_func=1 --run=ce_1 > run_ce_1.txt
python mnist_model_dcgan.py --num_epochs=50 --layer_list 512 256 --fclayer_list 2000 5000 --filter_sz=3 --batch_size=256 --loss_func=1 --run=ce_2 > run_ce_2.txt
python mnist_model_dcgan.py --num_epochs=50 --layer_list 512 256 --fclayer_list 2000 5000 --filter_sz=3 --lr=0.002 --batch_size=256 --loss_func=1 --run=ce_3 > run_ce_3.txt

