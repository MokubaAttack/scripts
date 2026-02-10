# reset_func
This is a function that deletes stablediffusion pipeline or stablediffusionxl pipeline. If you use this function, you can next diffusion that have different pipeline by restart the kernel.
## explanations
reset_func.reset_func( f, s )
- f : pipeline
- s : str ( default : "colab" ) In Colab, please input "colab". In Kaggle, please input "kaggle". This function reload system modules finally. If default system modules of colab or kaggle are reloaded, error is shown. Then you must decide which do you read colab_default.txt ( keys list of default system modules in colab ) or kaggle_default.txt ( keys list of default system modules in kaggle ).
