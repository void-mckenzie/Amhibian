import os
import json
base_config = {'exclude': ['heart', 'prostate_peripheral_T2', 'prostate_transitional_T2', 'spleen', 'colon'], 'params': {'tasks_per_iteration': 5, 'num_shots': 15, 'outer_epochs': 25, 'inner_epochs': 20, 'inner_lr': 0.01, 'meta_lr': 0.01, 'inner_wd': 0.003, 'meta_wd': 0.001, 'inter_epoch_decay': False, 'meta_decay': False, 'val_shots': 5, 'batch_size': 1, 'loss_type': 'iou', 'scheduler': 'exp', 'val_size': 0.1, 'weighted_update_type': "mean", 'scheduler_params': {'gamma': 0.8}, 'invert_weights': True, 'norm_type': 'instance', 'chart_name': None, "alpha": 0.2}, 'weights_name': 'weights_nonvolumetric_plain.pth'}

ft_config = {'weights_path': 'experiments/weights_nonvolumetric_plain.pth', 'targets': ['spleen'], 'report_path': 'out_results.csv', 'params': {'epochs': 20, 'batch_size': 1, 'ft_shots': 15, 'lr': 0.001, 'wd': 3e-05, 'loss_type': 'iou', 'scheduler': None, 'scheduler_params': None, 'norm_type': 'instance', 'volumetric': False, 'max_images': None, 'data_regime': 'all'}, 'exp_type': 'ft', 'num_selections': 1}

def write_json(filename, data):
	with open(filename, "w+") as f:
		json.dump(data, f, indent=4)

experiment_list = ["iou", "dice", "combined_dice_bce"]

experiment_config_dir = "experiment_section2_configs_amphibian"
os.makedirs(experiment_config_dir, exist_ok=True)
os.makedirs("log_dir_section2", exist_ok=True)

for loss_type in experiment_list:
	chart_name = f"Amphibian-U-Net-iw_mean-{loss_type}"
	experiment_config_file = f"{experiment_config_dir}/amphibian_meta_section2_{loss_type}.json"
	experiment_log_file = f"log_dir_section2/amphibian_meta_section2_{loss_type}.logs"
	config_copy = base_config.copy()
	config_copy["params"]["loss_type"] = loss_type
	config_copy["params"]["chart_name"] =  chart_name
	config_copy["weights_name"] = f"amphibian_weights_plain_section2_iw_mean_{loss_type}.pth"
	write_json(experiment_config_file, config_copy)
	ft_config_copy = ft_config.copy()
	weight_name = config_copy["weights_name"]
	ft_config_copy["weights_path"] = f"experiments/{weight_name}"
	ft_config_copy["params"]["loss_type"] = loss_type 
	ft_config_copy["report_path"] = f"amphibian_weights_plain_section2_{loss_type}.csv"
	ft_config_file = f"{experiment_config_dir}/amphibian_ft_section2_{loss_type}.json"
	ft_log_file = f"log_dir_section2/amphibian_ft_section2_{loss_type}.logs"
	write_json(ft_config_file, ft_config_copy)
	command1 = f"""(nohup python -u experiment_runner.py --exp_type=meta --params={experiment_config_file} > {experiment_log_file} && nohup python -u experiment_runner.py --exp_type=ft --params={ft_config_file} > {ft_log_file}) &
"""
	print(command1)

