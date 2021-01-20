# echo "ASP Train from scratch"
# python3.6 mnist_asp.py

echo "Pretrain"
python3.6 mnist_pretrain.py
echo "Finetune"
python3.6 mnist_finetune.py
