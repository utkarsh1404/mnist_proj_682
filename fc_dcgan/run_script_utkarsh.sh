#python mnist_model_FC.py --num_epochs=51 --loss_func=0 --run=fc_0
#python mnist_model_FC.py --num_epochs=51 --lr=0.00005 --loss_func=0 --run=fc_1
python mnist_model_FC.py --num_epochs=41 --layer_list 512 1024 2048 4096 8192 --batch_size=200 --loss_func=0 --run=fc_2
#python mnist_model_FC.py --num_epochs=41 --layer_list 512 1024 2048 4096 8192 --batch_size=200 --lr=0.00005 --loss_func=0 --run=fc_3
