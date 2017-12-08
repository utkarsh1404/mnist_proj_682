python mnist_model_dcgan.py --num_epochs=45 --loss_func=1 --run=ce_0
python mnist_model_dcgan.py --num_epochs=45 --lr=0.00005 --loss_func=1 --run=ce_1
python mnist_model_dcgan.py --num_epochs=45 --layer_list 256 128 --fclayer_list 1024 2048 --batch_size=200 --loss_func=1 --run=ce_2
#python mnist_model_dcgan.py --num_epochs=45 --layer_list 256 128 --fclayer_list 1024 2048 --lr=0.00005 --batch_size=200 --loss_func=1 --run=ce_3

