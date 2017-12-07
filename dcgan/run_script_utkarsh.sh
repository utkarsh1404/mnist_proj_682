python mnist_model_dcgan.py --num_epochs=51 --loss_func=0 --run=ls_0
python mnist_model_dcgan.py --num_epochs=51 --lr=0.00005 --loss_func=0 --run=ls_1
python mnist_model_dcgan.py --num_epochs=51 --layer_list 512 256 --fclayer_list 2000 5000 --filter_sz=3 --batch_size=200 --loss_func=0 --run=ls_2
#python mnist_model_dcgan.py --num_epochs=51 --layer_list 512 256 --fclayer_list 2000 5000 --filter_sz=3 --lr=0.00005 --batch_size=200 --loss_func=0 --run=ls_3

