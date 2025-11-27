#This file contains the default configuration setup referred to as "default setup" in the paper
default_config = {"epochs" : 15000, 
"steps_til_summary" : 250, 
"M" : [], 
"slice" : 175,
"hf_chunk_size": (96, 96, 4),
"lf_chunk_size": (96//2, 96//2, 4),
"l1" : 10000.0, 
"l2" : 0.0, #No L2 Norm
"l3" : 100, #seg_loss
"l4" :  "[0.95, 0.95, 0.95, 0.01]", #TV_seg
"l5" : "[1.5, 1.5, 1.5, 0.1]", #TV_img
"SIREN_FACTOR" : 30, #"w0" : 25 (old value)
"WIRE_OMEGA" : 20.0,
"WIRE_SIGMA" : 10.0,
"ffe" : True,
"in_features" : 3,
"lr" : 5e-5, 
"scheduler_step_size" : 250,
"scheduler_gamma" : 0.5,
"dataset_num" : 102, 
"activation_fn" : 'WIRE',
"is_new_contrast" : False,
"points_num" : 96*96*4,
"downsampled_points" : 48*48*4,
"sens_id" : -1,
"hidden_layers" : 5,
"hidden_features" : 128, 
"FF_FREQS" : 72,
"FF_SCALE" : 4,
}
