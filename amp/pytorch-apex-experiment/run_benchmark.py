import os

for mode in ['FP32','FP16','amp']:
	for batch_size in [128,256,512,1024,2048]:
		os.system("python CIFAR.py --GPU RTX --mode '"+ mode +"' --batch_size "+ str(batch_size) +" --iteration 100")

