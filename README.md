# Unparalleled Text summarization using GAN  
This is the official implementation of the paper [Learning to Encode Text as Human-Readable Summaries using Generative Adversarial Networks](https://arxiv.org/abs/1810.02851).  


## Dependencies  

* python3  
* tensorflow >= 1.0  


## Difference between original paper  
In this implementation, I use GAN training method proposed by [ScratchGAN](https://arxiv.org/pdf/1905.09922.pdf) for adversarial training. The performance is more robust and slightly better than original paper.  

## Running code:  
### Download English Gigaword:  
Download data from [Sent-Summary](https://github.com/harvardnlp/sent-summary). Then, move downloaded "train.article.txt" and "train.title.txt" to "giga_word" directory in this repository.  

### Make pretraining data:  
> python3 make_pretrain.py  

### Pretraining generator:  
> python3 main.py -pretrain -model_dir [model_path] -pretrain_input [input_path] -pretrain_target -num_steps 20000  

### Unparalleled Summarization Training:  
> python3 main.py -train -model_dir [model_path] -article [article_path] -summary [summary_path] -num_steps 7000  

The default setting can reproduce the results in the paper.

### Testing:  
> python3 main.py -test -model_dir [model_path] -test_input [input_path] -result_path [result_path]  
